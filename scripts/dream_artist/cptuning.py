import os
import sys
import traceback

import cv2
import numpy as np
import torch
import tqdm
import html
import datetime
import csv
from copy import deepcopy

from PIL import Image, PngImagePlugin
from torch.nn import functional as F
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
import math

from modules import shared, devices, sd_hijack, processing, sd_models, images
import scripts.dream_artist.dataset as DA_dataset
from modules.textual_inversion.learn_schedule import LearnRateScheduler

from modules.textual_inversion.image_embedding import (embedding_to_b64, embedding_from_b64,
                                                  insert_image_data_embed, extract_image_data_embed,
                                                  caption_image_overlay)
from .convnext_discriminator import XPDiscriminator
import json
from torchvision import transforms
import random

class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class EmbeddingDatabase:
    def __init__(self, embeddings_dir):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.dir_mtime = None
        self.embeddings_dir = embeddings_dir

    def register_embedding(self, embedding, model):

        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenizer([embedding.name], add_special_tokens=False)['input_ids'][0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def load_words_embeddings(self):
        mt = os.path.getmtime(self.embeddings_dir)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = []

            if os.path.splitext(filename.upper())[-1] in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
                embed_image = Image.open(path)
                if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                    data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                    name = data.get('name', name)
                else:
                    data = extract_image_data_embed(embed_image)
                    name = data.get('name', name)
            else:
                data = torch.load(path, map_location="cpu")

            # pseudo-words embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            # diffuser concepts
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)
            else:
                raise Exception(f"Couldn't identify {filename} as neither words embedding nor diffuser concept.")

            vec = emb.detach().to(devices.device, dtype=torch.float32)
            embedding = Embedding(vec, name)
            embedding.step = data.get('step', None)
            embedding.sd_checkpoint = data.get('sd_checkpoint', None)
            embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
            self.register_embedding(embedding, shared.sd_model)

        for fn in os.listdir(self.embeddings_dir):
            try:
                fullfn = os.path.join(self.embeddings_dir, fn)

                if os.stat(fullfn).st_size == 0:
                    continue

                process_file(fullfn, fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} words embeddings.")
        print("Embeddings:", ', '.join(self.word_embeddings.keys()))

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        current_params, ma_params = current_model.vec, ma_model.vec
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
    cond_model = shared.sd_model.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

    with devices.autocast():
        cond_model([""])  # will send cond model to GPU if lowvram/medvram is active

    ids = cond_model.tokenizer(init_text, max_length=num_vectors_per_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
    embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    for i in range(num_vectors_per_token):
        vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]
        if '-neg' in name:
            vec[i]+=torch.randn_like(vec[i])*1e-3

    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = Embedding(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn


def write_loss(log_directory, filename, step, epoch_len, values):
    if shared.opts.training_write_csv_every == 0:
        return

    if (step + 1) % shared.opts.training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        if write_csv_header:
            csv_writer.writeheader()

        epoch = step // epoch_len
        epoch_step = step % epoch_len

        csv_writer.writerow({
            "step": step + 1,
            "epoch": epoch,
            "epoch_step": epoch_step + 1,
            **values,
        })

def validate_train_inputs(model_name, learn_rate, batch_size, data_root, template_file, steps, save_model_every, create_image_every, log_directory, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert isinstance(batch_size, int), "Batch size must be integer"
    assert batch_size > 0, "Batch size must be positive"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_file, "Prompt template file is empty"
    assert os.path.isfile(template_file), "Prompt template file doesn't exist"
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0 , "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0 , "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0 , "Create image must be positive or 0"
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"

#hook DDPM p_losses to support negative prompt training and get output latent
from ldm.util import default
from ldm.modules.diffusionmodules.util import extract_into_tensor

#a_t=0.005
#sqrt_one_minus_at=np.sqrt(1.-a_t)
def p_losses_hook(x_start, cond, t, noise=None, scale=(1.0,1.0), att_mask=None, dy_cfg_f='ln'):
    self=shared.sd_model
    noise = default(noise, lambda: torch.randn_like(x_start))
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

    # support negative prompt tuning
    t_raw = t
    x_noisy_raw = x_noisy
    if scale[1] != 1.0:
        x_noisy = torch.cat([x_noisy] * 2)
        t = torch.cat([t] * 2)

    model_output = self.apply_model(x_noisy, t, cond)

    # support negative prompt tuning
    if scale[1] != 1.0:
        e_t_uncond, e_t = model_output.chunk(2)
        if scale[0] != scale[1]:
            rate = t_raw / (self.num_timesteps - 1)
            if dy_cfg_f=='cos':
                rate = torch.cos((rate-1)*math.pi/2)
            elif dy_cfg_f=='cos2':
                rate = 1-torch.cos(rate*math.pi/2)
            elif dy_cfg_f=='ln':
                pass
            else:
                rate = eval(dy_cfg_f)
        else:
            rate = 1
        model_output = e_t_uncond + ((scale[1]-scale[0])*rate+scale[0]) * (e_t - e_t_uncond)

    loss_dict = {}
    prefix = 'train' if self.training else 'val'

    if self.parameterization == "x0":
        target = x_start
    elif self.parameterization == "eps":
        target = noise
    else:
        raise NotImplementedError()

    loss_simple = self.get_loss(model_output, target, mean=False)
    if att_mask is not None:
        loss_simple=loss_simple*att_mask
    loss_simple=loss_simple.mean([1, 2, 3])
    loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

    logvar_t = self.logvar[t_raw].to(self.device)
    loss = loss_simple / torch.exp(logvar_t) + logvar_t
    # loss = loss_simple / torch.exp(self.logvar) + self.logvar
    if self.learn_logvar:
        loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
        loss_dict.update({'logvar': self.logvar.data.mean()})

    loss = self.l_simple_weight * loss.mean()

    loss_vlb = self.get_loss(model_output, target, mean=False)
    if att_mask is not None:
        loss_vlb=loss_vlb*att_mask
    loss_vlb=loss_vlb.mean(dim=(1, 2, 3))
    loss_vlb = (self.lvlb_weights[t_raw] * loss_vlb).mean()
    loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
    loss += (self.original_elbo_weight * loss_vlb)
    loss_dict.update({f'{prefix}/loss': loss})

    img = (x_noisy_raw-extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t_raw, x_start.shape) * model_output)/extract_into_tensor(self.sqrt_alphas_cumprod, t_raw, x_start.shape)

    return loss, loss_dict, img

def get_cfg_range(cfg_text:str):
    dy_cfg_f='ln'
    if cfg_text.find(':')!=-1:
        cfg_text, dy_cfg_f = cfg_text.split(':')

    if cfg_text.find('-')!=-1:
        l, h = cfg_text.split('-')
        return float(l), float(h), dy_cfg_f
    else:
        return float(cfg_text), float(cfg_text), dy_cfg_f

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms

def train_embedding(_, embedding_name, seed, learn_rate, batch_size, data_root, log_directory, training_width, training_height, steps, create_image_every, save_embedding_every, template_file, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height,
                    cfg_scale, classifier_path, use_negative, use_att_map, use_rec, rec_loss_w, neg_lr_w, ema_w, ema_rep_step, ema_w_neg, ema_rep_step_neg, adam_beta1, adam_beta2, fw_pos_only, accumulation_steps,
                    unet_train, unet_lr):
    # NOTE: the first `_` argument is the TaskID, which is not used in this function
    # but is required by Automatic1111 to set the TaskID for the thread.
    set_seed(seed)

    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    validate_train_inputs(embedding_name, learn_rate, batch_size, data_root, template_file, steps, save_embedding_every, create_image_every, log_directory, name="embedding")

    p_losses_backup = shared.sd_model.p_losses
    shared.sd_model.p_losses = p_losses_hook  # hook p_losses

    #maybe fix issue #1
    shared.sd_model.first_stage_model.to(devices.device)

    shared.state.textinfo = "Initializing prompt tuning..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    unload = False #shared.opts.unload_models_when_training

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    cond_model = shared.sd_model.cond_stage_model

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    checkpoint = sd_models.select_checkpoint()

    ititial_step = embedding.step or 0
    if ititial_step >= steps:
        shared.state.textinfo = f"Model has already been trained beyond specified max steps"
        return embedding, filename

    scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)

    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = DA_dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=shared.sd_model, device=devices.device, template_file=template_file, batch_size=batch_size, fw_pos_only=fw_pos_only)
    if unload:
        shared.sd_model.first_stage_model.to(devices.cpu)

    ema = EMA(ema_w)
    ema_neg = EMA(ema_w_neg)

    embedding_ema = deepcopy(embedding)

    embedding.vec.requires_grad = True
    if use_negative:
        embedding_neg = hijack.embedding_db.word_embeddings[embedding_name + '-neg']  # negative prompt embeddings
        embedding_neg_ema = deepcopy(embedding_neg)

        embedding_neg.vec.requires_grad = True

    hyper_param = {
        'lr': learn_rate,
        'bs': batch_size,
        'cfg': cfg_scale,
        'size': [training_width, training_height],
        'neg': use_negative,
        'rec': use_rec,
        'seed': seed,
        'prompt_len': embedding.vec.shape,
    }
    if use_negative:
        hyper_param['prompt_len_neg'] = embedding_neg.vec.shape
        hyper_param['neg_lr_w'] = neg_lr_w
    if use_rec:
        hyper_param['rec_loss_w'] = rec_loss_w

    hyper_param = json.dumps(hyper_param, sort_keys=True, indent=4)
    with open(os.path.join(log_directory, 'hyper_param.json'), 'w') as f:
        f.write(hyper_param)

    cfg_l, cfg_h, dy_cfg_f = get_cfg_range(cfg_scale)

    disc = XPDiscriminator(classifier_path) if (classifier_path is not None) and os.path.exists(classifier_path) else None

    if disc is not None:
        print('use convnext discriminator')

    unet = shared.sd_model.model.diffusion_model
    unet_down = unet.input_blocks
    unet_up = unet.output_blocks
    #,print(shared.sd_model.model.diffusion_model)

    def get_convs(block):
        return block[1].norm, block[1].proj_in, block[1].proj_out

    unet_part_list = [
        unet_down[1],
        unet_down[2][0], *get_convs(unet_down[2]), unet_down[3],
        unet_down[4][0], *get_convs(unet_down[4]), unet_down[5][0], *get_convs(unet_down[5]), unet_down[6],

        unet_up[8][0], *get_convs(unet_up[8]), unet_up[8][2], unet_up[7][0], *get_convs(unet_up[7]),
        unet_up[10][0], *get_convs(unet_up[10]), unet_up[9][0], *get_convs(unet_up[9]),
        unet_up[11],
    ]

    if unet_train:
        for layer in unet_part_list:
            layer.requires_grad_(True)
        unet.train()
    unet_lr = float(unet_lr)

    num_warmup_steps = 100
    num_training_steps = steps
    num_cycles = 0.5
    rate_min = 0.1
    def lr_lambda_cos(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        rate = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return max(0.0, rate_min+rate*(1-rate_min))

    if use_negative:
        #optimizer = torch.optim.AdamW([embedding.vec, embedding_neg.vec], lr=scheduler.learn_rate)
        optimizer = torch.optim.AdamW([
                {'params': embedding.vec},
                {'params': embedding_neg.vec, 'lr': scheduler.learn_rate*neg_lr_w},
            ], lr=scheduler.learn_rate, betas=(adam_beta1, adam_beta2))
        optimizer_unet = torch.optim.AdamW([
            {'params': layer.parameters(), 'initial_lr': unet_lr} for layer in unet_part_list
        ], lr=unet_lr, eps=1e-6)
        #scheduler_unet = get_constant_schedule_with_warmup(optimizer_unet, num_warmup_steps=100, last_epoch=ititial_step)
        scheduler_unet = LambdaLR(optimizer_unet, lr_lambda_cos, ititial_step)
    else:
        optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate)


    losses = torch.zeros((32,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False

    pbar = tqdm.tqdm(enumerate(ds), total=steps-ititial_step)
    for i, entries in pbar:
        embedding.step = i + ititial_step
        if use_negative:
            embedding_neg.step = i + ititial_step

        scheduler.apply(optimizer, embedding.step)
        if scheduler.finished:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            #c = cond_model([entry.cond_text for entry in entries])
            if use_negative:
                #uc = cond_model([entry.cond_text_neg.replace(ds.placeholder_token, ds.placeholder_token+'-neg') for entry in entries])
                c_in = cond_model([entry.cond_text_neg.replace(ds.placeholder_token, ds.placeholder_token+'-neg') for entry in entries]+
                                  [entry.cond_text for entry in entries])
            else:
                c_in = cond_model([entry.cond_text for entry in entries])

            x = torch.stack([entry.latent for entry in entries]).to(devices.device)

            if use_att_map:
                att_mask = torch.stack([(entry.att_mask if entry.att_mask is not None else torch.ones_like(entry.latent)) for entry in entries]).to(devices.device)
                output = shared.sd_model(x, c_in, scale=(cfg_l, cfg_h), att_mask=att_mask, dy_cfg_f=dy_cfg_f)
            else:
                output = shared.sd_model(x, c_in, scale=(cfg_l, cfg_h), att_mask=None, dy_cfg_f=dy_cfg_f)

            if disc is not None or use_rec:
                x_samples_ddim = shared.sd_model.decode_first_stage.__wrapped__(shared.sd_model, output[2])  # forward with grad

            if disc is not None:
                # loss = ce(disc.get_all(x_samples_ddim), disc_label)
                loss = (1 - disc.get_score(x_samples_ddim)).mean()
            elif use_rec:
                loss = output[0] + F.l1_loss(torch.cat([entry.timg for entry in entries]), x_samples_ddim) * rec_loss_w
            else:
                loss = output[0]
            del x

            losses[embedding.step % losses.shape[0]] = loss.item()
            loss = loss / accumulation_steps

            if (i + 2) % accumulation_steps == 0:
                optimizer.zero_grad()
                if unet_train:
                    optimizer_unet.zero_grad()

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if unet_train:
                    for layer in unet_part_list:
                        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1)
                    optimizer_unet.step()
                    scheduler_unet.step()

                torch.nn.utils.clip_grad_norm_(embedding.vec, 1)
                torch.nn.utils.clip_grad_norm_(embedding_neg.vec, 1)
                optimizer.step()

            with torch.no_grad():
                if ema_w != 1:
                    ema.update_model_average(embedding_ema, embedding)
                    if (i+1)%ema_rep_step == 0:
                        embedding.vec.data = deepcopy(embedding_ema.vec.data)

                if ema_w_neg != 1:
                    ema_neg.update_model_average(embedding_neg_ema, embedding_neg)
                    if (i + 1) % ema_rep_step_neg == 0:
                        embedding_neg.vec.data = deepcopy(embedding_neg_ema.vec.data)

        steps_done = embedding.step + 1

        ds_len = ds.batch_size
        epoch_num = embedding.step // ds_len
        epoch_step = embedding.step % ds_len

        pbar.set_description(f"[Epoch {epoch_num}: {epoch_step}/{ds_len}]loss: {losses.mean():.7f}, "
                             f"grad:{embedding.vec.grad.detach().cpu().abs().mean().item():.7f}, "
                             f"grad_neg:{embedding_neg.vec.grad.detach().cpu().abs().mean().item() if use_negative else 0:.7f}, "
                             f"lr_unet:{optimizer_unet.state_dict()['param_groups'][0]['lr']}")

        if embedding_dir is not None and steps_done % save_embedding_every == 0:
            # Before saving, change name to match current checkpoint.
            embedding_name_every = f'{embedding_name}-{steps_done}'
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
            save_embedding(embedding, checkpoint, embedding_name_every, last_saved_file, remove_cached_checksum=True,
                           use_negative=use_negative, embedding_neg=embedding_neg, unet_layers=unet_part_list)
            embedding_yet_to_be_embedded = True

        write_loss(log_directory, "prompt_tuning_loss.csv", embedding.step, ds_len, {
            "loss": f"{losses.mean():.7f}",
            "learn_rate": scheduler.learn_rate
        })

        if images_dir is not None and steps_done % create_image_every == 0:
            forced_filename = f'{embedding_name}-{steps_done}'
            last_saved_image = os.path.join(images_dir, forced_filename)

            shared.sd_model.first_stage_model.to(devices.device)

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=preview_prompt,
                do_not_save_grid=True,
                do_not_save_samples=True,
                do_not_reload_embeddings=True,
                negative_prompt=preview_prompt.replace(ds.placeholder_token, ds.placeholder_token + '-neg') if use_negative else None,
                cfg_scale=float(cfg_scale) if use_negative else 1.0,
            )

            if preview_from_txt2img:
                p.prompt = preview_prompt
                p.negative_prompt = preview_negative_prompt

                p.steps = preview_steps
                p.sampler_index = preview_sampler_index
                p.cfg_scale = preview_cfg_scale
                p.seed = preview_seed
                p.width = preview_width
                p.height = preview_height
            else:
                p.prompt = entries[0].cond_text
                if use_negative:
                    p.negative_prompt = entries[0].cond_text_neg.replace(ds.placeholder_token, ds.placeholder_token + '-neg')
                    p.cfg_scale = float(cfg_scale)
                p.steps = 20
                p.width = training_width
                p.height = training_height

            preview_text = p.prompt

            processed = processing.process_images(p)
            image = processed.images[0]

            if unload:
                shared.sd_model.first_stage_model.to(devices.cpu)

            shared.state.current_image = image

            if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:

                last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

                info = PngImagePlugin.PngInfo()
                data = torch.load(last_saved_file)
                info.add_text("sd-ti-embedding", embedding_to_b64(data))

                title = "<{}>".format(data.get('name', '???'))

                try:
                    vectorSize = list(data['string_to_param'].values())[0].shape[0]
                except Exception as e:
                    vectorSize = '?'

                checkpoint = sd_models.select_checkpoint()
                footer_left = checkpoint.model_name
                footer_mid = '[{}]'.format(checkpoint.hash)
                footer_right = '{}v {}s'.format(vectorSize, steps_done)

                captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                captioned_image = insert_image_data_embed(captioned_image, data)

                captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                embedding_yet_to_be_embedded = False

            last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
            last_saved_image += f", prompt: {preview_text}"

            #set seed, seed is change by p
            seed+=1
            set_seed(seed)

        shared.state.job_no = embedding.step

        shared.state.textinfo = f"""
<p>
Loss: {losses.mean():.7f}<br/>
Step: {embedding.step} (Accumulation: {((i + 1) % accumulation_steps) + 1})<br/>
Last prompt: {html.escape(entries[0].cond_text)}<br/>
Last negative prompt: {html.escape(entries[0].cond_text_neg.replace(ds.placeholder_token, ds.placeholder_token+'-neg'))}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
    save_embedding(embedding, checkpoint, embedding_name, filename, remove_cached_checksum=True, use_negative=use_negative, embedding_neg=embedding_neg,
                   unet_layers=unet_part_list)
    shared.sd_model.first_stage_model.to(devices.device)

    shared.sd_model.p_losses = p_losses_backup
    for layer in unet_part_list:
        layer.requires_grad_(True)
    unet.eval()

    return embedding, filename

def save_embedding(embedding, checkpoint, embedding_name, filename, remove_cached_checksum=True, use_negative=False, embedding_neg=None, unet_layers=None):
    old_embedding_name = embedding.name
    old_sd_checkpoint = embedding.sd_checkpoint if hasattr(embedding, "sd_checkpoint") else None
    old_sd_checkpoint_name = embedding.sd_checkpoint_name if hasattr(embedding, "sd_checkpoint_name") else None
    old_cached_checksum = embedding.cached_checksum if hasattr(embedding, "cached_checksum") else None
    try:
        embedding.sd_checkpoint = checkpoint.hash
        embedding.sd_checkpoint_name = checkpoint.model_name
        if remove_cached_checksum:
            embedding.cached_checksum = None
        embedding.name = embedding_name
        embedding.save(filename)

        if use_negative:
            embedding_neg.sd_checkpoint = checkpoint.hash
            embedding_neg.sd_checkpoint_name = checkpoint.model_name
            if remove_cached_checksum:
                embedding_neg.cached_checksum = None
            embedding_neg.name = embedding_name+'-neg'
            embedding_neg.save(f'{filename[:-3]}-neg.pt')

    except:
        embedding.sd_checkpoint = old_sd_checkpoint
        embedding.sd_checkpoint_name = old_sd_checkpoint_name
        embedding.name = old_embedding_name
        embedding.cached_checksum = old_cached_checksum

        if use_negative:
            embedding_neg.sd_checkpoint = old_sd_checkpoint
            embedding_neg.sd_checkpoint_name = old_sd_checkpoint_name
            embedding_neg.name = old_embedding_name+'-neg'
            embedding_neg.cached_checksum = old_cached_checksum

        raise

def proc_att(data_root, training_width, training_height):
    shared.sd_model.first_stage_model.to(devices.device)

    shared.state.textinfo = "Processing Att-Map"
    shared.state.job_count = 0

    att_map=[None]
    att_proc=[None]

    with torch.autocast("cuda"):
        ds = DA_dataset.DataAtt(data_root=data_root, width=training_width, height=training_height)

    def att_hook(module, x, output):
        h_ = x[0]
        h_ = module.norm(h_)
        q = module.q(h_)
        k = module.k(h_)
        v = att_map[0]

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, 1, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        proc = (torch.bmm(v, w_)-v.mean() + v).clamp(0.,1.)
        att_proc[0] = proc.view(b,1,h,w)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        #att_proc[0] = v.view(b, 1, h, w)

    hook = shared.sd_model.first_stage_model.encoder.mid.attn_1.register_forward_hook(att_hook)

    pbar = tqdm.tqdm(enumerate(ds), total=len(ds))
    import torch.nn.functional as F
    for i, entries in pbar:
        with torch.autocast("cuda"):
            timg = torch.cat([entry.timg for entry in entries]).to(devices.device)
            att_map[0]=torch.stack([(entry.att_mask if entry.att_mask is not None else torch.ones_like(entry.latent)) for entry in entries]).to(devices.device)
            #att_map[0] = F.interpolate(timg, scale_factor=1/8, mode='bicubic')
            shared.sd_model.encode_first_stage(timg)

            att_map_new = att_proc[0].detach().cpu().squeeze(0)#.permute(1,2,0).numpy().astype(np.uint8)
            #att_map_new = cv2.resize(att_map_new, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            att_map_new = transforms.ToPILImage()(att_map_new)
            #att_map_new = att_map_new.resize((int(att_map_new.size[0]*8), int(att_map_new.size[1]*8)), PIL.Image.BICUBIC)
            att_map_new.save(entries[0].filename[:entries[0].filename.rfind('.')]+'_att_proc'+entries[0].filename[entries[0].filename.rfind('.'):])

        shared.state.job_no = i

    hook.remove()