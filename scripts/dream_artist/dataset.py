import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random
import tqdm
from modules import devices, shared
import re

re_numbers_at_start = re.compile(r"^[-\d]+\s*")


class DatasetEntry:
    def __init__(self, filename=None, latent=None, filename_text=None, timg=None, att_mask=None):
        self.filename = filename
        self.latent = latent
        self.att_mask = att_mask
        self.filename_text = filename_text
        self.timg = timg
        self.cond = None
        self.cond_neg = None
        self.cond_text = None
        self.cond_text_neg = None

class RatioCrop():
    def __init__(self, width, height):
        self.width=width
        self.height=height

    def __call__(self, image):
        w, h = image.size
        return image.crop((w//2-self.width//2, h//2-self.height//2, w//2+self.width//2, h//2+self.height//2))


class PersonalizedBase(Dataset):
    def __init__(self, data_root, width, height, repeats, flip_p=0.5, placeholder_token="*", model=None, device=None, template_file=None, include_cond=False, batch_size=1, fw_pos_only=False):
        re_word = re.compile(shared.opts.dataset_filename_word_regex) if len(shared.opts.dataset_filename_word_regex) > 0 else None

        self.placeholder_token = placeholder_token

        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.dataset = []

        with open(template_file, "r") as file:
            lines = [x.strip() for x in file.readlines()]

        self.flag_filewords=False
        if template_file.find('filewords')!=-1:
            if fw_pos_only:
                self.flag_filewords = True
            idx_fw = template_file.find('_filewords')
            template_file_nofw = template_file[:idx_fw]+template_file[idx_fw+len('_filewords'):]
            with open(template_file_nofw, "r") as file:
                lines_nofw = [x.strip() for x in file.readlines()]
            self.lines_nofw = lines_nofw

        self.lines = lines

        assert data_root, 'dataset directory not specified'
        assert os.path.isdir(data_root), "Dataset directory doesn't exist"
        assert os.listdir(data_root), "Dataset directory is empty"

        # rasize and crop, not change the image ratio
        TR = transforms.Compose([
            transforms.Resize(min(self.width, self.height), interpolation=transforms.InterpolationMode.BICUBIC),
            #transforms.CenterCrop(min(self.width, self.height)),
            RatioCrop(self.width, self.height),
            transforms.ToTensor()
        ])

        cond_model = shared.sd_model.cond_stage_model

        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
        print("Preparing dataset...")
        for path in tqdm.tqdm(self.image_paths):
            if path[:path.rfind('.')].endswith('_att'):
                continue
            try:
                image = Image.open(path).convert('RGB')
            except Exception:
                continue

            text_filename = os.path.splitext(path)[0] + ".txt"
            filename = os.path.basename(path)

            if os.path.exists(text_filename):
                with open(text_filename, "r", encoding="utf8") as file:
                    filename_text = file.read()
            else:
                filename_text = os.path.splitext(filename)[0]
                filename_text = re.sub(re_numbers_at_start, '', filename_text)
                if re_word:
                    tokens = re_word.findall(filename_text)
                    filename_text = (shared.opts.dataset_filename_join_string or "").join(tokens)

            torchdata = (TR(image) * 2. - 1.).to(device=device, dtype=torch.float32)

            timg = torchdata.unsqueeze(dim=0)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(timg)).squeeze()
            init_latent = init_latent.to(devices.cpu)

            entry = DatasetEntry(filename=path, filename_text=filename_text, latent=init_latent, timg=timg)

            if include_cond:
                sidx = random.randint(0, len(self.lines) - 1)

                entry.cond_text = self.create_text(filename_text, idx=sidx)
                entry.cond = cond_model([entry.cond_text]).to(devices.cpu).squeeze(0)

                if self.flag_filewords:
                    entry.cond_text_neg = self.create_text_nofw(idx=sidx)
                    entry.cond_neg = cond_model([entry.cond_text_neg]).to(devices.cpu).squeeze(0)
                else:
                    entry.cond_text_neg = self.create_text(filename_text, idx=sidx)
                    entry.cond_neg = cond_model([entry.cond_text_neg]).to(devices.cpu).squeeze(0)

            att_path=path[:path.rfind('.')] + '_att' + path[path.rfind('.'):]
            if os.path.exists(att_path):
                print(att_path)
                att_mask = Image.open(att_path).convert('L').resize((self.width//8, self.height//8), PIL.Image.BICUBIC)
                np_mask = np.array(att_mask).astype(float)[:,:,None]
                np_mask[np_mask<=127+0.1]=(np_mask[np_mask<=127+0.1]/127.)#*0.99+0.01
                np_mask[np_mask>127]=((np_mask[np_mask>127]-127)/128.)*4+1

                torchdata = torch.from_numpy(np_mask).to(device=device, dtype=torch.float32)
                torchdata = torch.moveaxis(torchdata, 2, 0)
                entry.att_mask = torchdata

            self.dataset.append(entry)

        assert len(self.dataset) > 0, "No images have been found in the dataset."
        self.length = int(np.ceil(len(self.dataset) * repeats / batch_size))

        self.dataset_length = len(self.dataset)
        self.indexes = None
        self.shuffle()

    def shuffle(self):
        self.indexes = np.random.permutation(self.dataset_length)

    def create_text(self, filename_text, idx=None):
        text = random.choice(self.lines) if idx is None else self.lines[idx]
        text = text.replace("[name]", self.placeholder_token)
        text = text.replace("[filewords]", filename_text)
        return text

    def create_text_nofw(self, idx=None):
        text = random.choice(self.lines_nofw) if idx is None else self.lines_nofw[idx]
        text = text.replace("[name]", self.placeholder_token)
        return text

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        res = []

        for j in range(self.batch_size):
            position = i * self.batch_size + j
            if position % len(self.indexes) == 0:
                self.shuffle()

            index = self.indexes[position % len(self.indexes)]
            entry = self.dataset[index]

            if entry.cond is None:
                sidx = random.randint(0, len(self.lines) - 1)

                entry.cond_text = self.create_text(entry.filename_text, idx=sidx)
                if self.flag_filewords:
                    entry.cond_text_neg = self.create_text_nofw(idx=sidx)
                else:
                    entry.cond_text_neg = self.create_text(entry.filename_text, idx=sidx)

            res.append(entry)

        return res

class DataAtt(Dataset):
    def __init__(self, data_root, width, height, device=None):
        re_word = re.compile(shared.opts.dataset_filename_word_regex) if len(shared.opts.dataset_filename_word_regex) > 0 else None

        self.width = width
        self.height = height
        self.batch_size = 1

        self.dataset = []

        assert data_root, 'dataset directory not specified'
        assert os.path.isdir(data_root), "Dataset directory doesn't exist"
        assert os.listdir(data_root), "Dataset directory is empty"

        # rasize and crop, not change the image ratio
        TR = transforms.Compose([
            transforms.Resize(min(self.width, self.height), interpolation=transforms.InterpolationMode.BICUBIC),
            #transforms.CenterCrop(min(self.width, self.height)),
            RatioCrop(self.width, self.height),
            transforms.ToTensor()
        ])

        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
        print("Preparing dataset...")
        for path in tqdm.tqdm(self.image_paths):
            if path[:path.rfind('.')].endswith('_att'):
                continue
            try:
                image = Image.open(path).convert('RGB')
            except Exception:
                continue

            filename = os.path.basename(path)

            torchdata = (TR(image) * 2. - 1.).to(device=device, dtype=torch.float32)

            timg = torchdata.unsqueeze(dim=0)

            entry = DatasetEntry(filename=path, timg=timg)

            att_path=path[:path.rfind('.')] + '_att' + path[path.rfind('.'):]
            if os.path.exists(att_path):
                print(att_path)
                att_mask = Image.open(att_path).convert('L').resize((self.width//8, self.height//8), PIL.Image.BICUBIC)

                torchdata = transforms.ToTensor()(att_mask)
                entry.att_mask = torchdata

                self.dataset.append(entry)

        assert len(self.dataset) > 0, "No images have been found in the dataset."
        self.length = len(self.dataset)

        self.dataset_length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        res = []

        for j in range(self.batch_size):
            position = i * self.batch_size + j
            entry = self.dataset[position]

            res.append(entry)

        return res