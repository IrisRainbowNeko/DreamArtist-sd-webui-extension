# DreamArtist (webui Eextension)
This repo is the official ***Stable-Diffusion-webui extension version** implementation of ***"DreamArtist: Towards Controllable One-Shot Text-to-Image Generation via Contrastive Prompt-Tuning"*** 
with [Stable-Diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

Regular version: [DreamArtist](https://github.com/7eu7d7/DreamArtist-stable-diffusion)

Everyone is an artist. Rome wasn't built in a day, but your artist dreams can be!

With just ***one*** training image DreamArtist learns the content and style in it, generating diverse high-quality images with high controllability.
Embeddings of DreamArtist can be easily combined with additional descriptions, as well as two learned embeddings.

![](imgs/exp1.jpg)
![](imgs/exp_text1.jpg)
![](imgs/exp_text2.jpg)
![](imgs/exp_text3.jpg)

# Setup and Running

Clone this repo to extension folder.
```bash
git clone https://github.com/7eu7d7/DreamArtist-sd-webui-extension.git extensions/DreamArtist
```

## Training

First create the positive and negative embeddings in ```DreamArtist Create Embedding``` Tab.
![](imgs/create.jpg)

Then, select positive embedding and set the parameters and image folder path in the ```DreamArtist Train``` Tab to start training.
The corresponding negative embedding is loaded automatically.
If your VRAM is low or you want save time, you can uncheck the ```reconstruction```.
![](imgs/train.jpg)


## Tested models (need ema version):
+ Stable Diffusion v1.5
+ animefull-latest
+ Anything v3.0

# Style Clone
![](imgs/exp_style.jpg)

# Prompt Compositions
![](imgs/exp_comp.jpg)

# Comparison on One-Shot Learning
![](imgs/cmp.jpg)

# Other Results
![](imgs/cnx.jpg)
![](imgs/cnx2.jpg)

