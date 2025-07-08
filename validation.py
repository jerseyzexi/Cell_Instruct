import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets.features import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from perturbation_encoder import PerturbationEncoder, PerturbationEncoderInference
from transformers import CLIPTextModel, CLIPTokenizer, AutoFeatureExtractor
import sys
import diffusers
import os
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, \
    StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from PIL import ImageFile
from typing import Optional

logger = get_logger(__name__, log_level="INFO")

perturbation_encoder = PerturbationEncoder(
    dataset_id="HUVEC",
    model_type="conditional",
    model_name="SD",
)

class CustomInstructPix2PixPipeline(StableDiffusionInstructPix2PixPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        unet,
        scheduler,
        feature_extractor,
        safety_checker=None,
        tokenizer=None,
        custom_text_encoder=None,      # 新增一个 slot，用来接收你的自定义编码器
        image_encoder=None,
        requires_safety_checker=False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        # 如果你传了 custom_text_encoder，就用它；否则退回到原 text_encoder
        self.custom_text_encoder = custom_text_encoder or self.text_encoder
    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
    ):

        embeddings = self.custom_text_encoder(prompt)
        print(f"embediings: {embeddings.shape}")
        embeddings = embeddings.to(device)
        return embeddings




def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)
def preprocess_images(example):
    orig_np = convert_to_np(example["original_image"], 256)  # shape (3, H, W)
    # 2. 处理编辑图
    edit_np = convert_to_np(example["edited_image"], 256)  # shape (3, H, W)

    # 3. 合并成一个数组，维度 (2, 3, H, W)：第 0 维表示原图 vs 编辑图
    imgs = np.stack([orig_np, edit_np], axis=0)

    # 4. 转成 Tensor 并归一化到 [-1, 1]
    imgs_t = torch.from_numpy(imgs).float()  # (2, 3, H, W)
    imgs_t = 2 * (imgs_t / 255.0) - 1  # 归一化

    return imgs_t  # 返回一个 Tensor


def preprocess_train(example):
    # Preprocess images.
    imgs_t = preprocess_images(example)
    # Since the original and edited images were concatenated before
    # applying the transformations, we need to separate them and reshape
    # them accordingly.
    orig_t, edit_t = imgs_t[0], imgs_t[1]  # each shape=(3,256,256)

    # 3) 放回 examples
    example["original_pixel_values"] = orig_t  # Tensor shape (3,256,256)
    example["edited_pixel_values"] = edit_t  # same

    # 4) 处理 prompt，直接给单个字符串做 embedding
    prompt = example["edit_prompt"]  # e.g. "add fluorescence marker"

    example["input_ids"] = prompt  # Tensor shape (embed_dim,)

    return example


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--num_validation_images",
        type=str,
        default="/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

def safe_filter(example):
    try:
        # 这里尝试打开两张图，如果有任何一张损坏就会触发 OSError
        _ = example["original_image"].convert("RGB")
        _ = example["edited_image"].convert("RGB")
        return True
    except OSError:
        return False

def spy(name):
    def _hook(mod, ins, outs):
        if outs[0].dim() == 3:           # (B*N , S , C)
            C = outs[0].shape[-1]
        else:                            # (B*N , C)
            C = outs[0].shape[-1]
        if C == 3072:
            print(">>> culprit:", name, outs[0].shape)
            # 抛出一个有意义的异常，方便你定位
            raise RuntimeError(f"Found shape 3072 in {name}")
    return _hook

if __name__ == "__main__":

    print("This process is:", sys.executable)
    args = parse_args()
    accelerator = Accelerator()
    print("Accelerator device:", accelerator.device)
    dataset = load_dataset(
        "csv",
        data_files="validate.csv",
    )
    dataset = dataset.cast_column("original_image", Image())
    dataset = dataset.cast_column("edited_image", Image())
    dataset["train"] = dataset["train"].filter(safe_filter)
    train_dataset = dataset["train"].with_transform(preprocess_train)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    custom_text_encoder = PerturbationEncoderInference(
        dataset_id="HUVEC",
        model_type="conditional",
        model_name="SD")

    input_dir = "/root/autodl-tmp/instruct-pix2pix-model/checkpoint-10000/"
    unet = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet").to(accelerator.device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="feature_extractor"
    )
    vae = AutoencoderKL.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="vae")
    pipeline =CustomInstructPix2PixPipeline(
        vae = vae,
        unet=unwrap_model(unet),
        text_encoder=custom_text_encoder,
        feature_extractor=feature_extractor,
        scheduler=noise_scheduler
    )

    for n, m in pipeline.unet.named_modules():
        if n.endswith("attn2"):  # 只盯交叉注意力
            m.register_forward_hook(spy(n))

    df = pd.read_csv(r"/root/autodl-tmp/validate.csv")

    pipeline = pipeline.to(accelerator.device)
    edited_images = []
    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    pre_url = ""

    for idx in range(min(10, len(dataset))):
        sample = dataset["train"][idx]
        sample =preprocess_train(sample)
        # 如果你在 preprocess_train 中把原始像素存到了 original_pixel_values
        original_image = sample["original_pixel_values"]  # Tensor, shape=(3, H, W)
        edited = sample["edited_pixel_values"]  # Tensor, shape=(3, H, W)
        prompt = sample["input_ids"]
        print(f"origin_image:{original_image.shape}")
        print("prompt:"+prompt)
        edited_images.append(
            pipeline(
                prompt,
                image=original_image,
                num_inference_steps=20,
                image_guidance_scale=1.5,
                guidance_scale=7,
                generator=generator,
            ).images[0]
        )

output_dir = "edited_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 2. 遍历列表并保存
for idx, img in enumerate(edited_images, start=1):
    # 格式化文件名，比如 edited_001.png、edited_002.png…
    filename = f"edited_{idx:03d}.png"
    filepath = os.path.join(output_dir, filename)

    # img 是 PIL.Image.Image 对象，直接调用 save()
    img.save(filepath)

print(f"Saved {len(edited_images)} images to '{output_dir}/'")