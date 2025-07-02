from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Iterable, List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
import pandas as pd

DEFAULT_BASE_PATH = 'gs://rxrx1-us-central1'
DEFAULT_METADATA_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'metadata')
DEFAULT_IMAGES_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'images')
DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
RGB_MAP: Dict[int, Dict[str, torch.Tensor]] = {
    1: {'rgb': torch.tensor([19, 0, 249]), 'range': [0, 51]},
    2: {'rgb': torch.tensor([42, 255, 31]), 'range': [0, 107]},
    3: {'rgb': torch.tensor([255, 0, 25]), 'range': [0, 64]},
    4: {'rgb': torch.tensor([45, 255, 252]), 'range': [0, 191]},
    5: {'rgb': torch.tensor([250, 0, 253]), 'range': [0, 89]},
    6: {'rgb': torch.tensor([254, 255, 40]), 'range': [0, 191]},
}


def load_image(image_path: str) -> torch.Tensor:
    """Load a single-channel image as a tensor of shape ``(H, W)``."""
    img = read_image(image_path)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    return img


def load_images_as_tensor(image_paths: Iterable[str], dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """Load multiple channel images and stack them into a tensor of shape ``(H, W, N)``."""
    imgs: List[torch.Tensor] = [load_image(p).to(dtype) for p in image_paths]
    tensor = torch.stack(imgs, dim=0).permute(1, 2, 0)
    return tensor


def convert_tensor_to_rgb(t: torch.Tensor,
                          channels: Tuple[int, ...] = DEFAULT_CHANNELS,
                          vmax: int = 255,
                          rgb_map: Dict[int, Dict[str, torch.Tensor]] = RGB_MAP) -> torch.Tensor:
    """Convert an ``(H, W, N)`` multi-channel tensor to an RGB tensor."""
    colored = []
    for i, channel in enumerate(channels):
        ch = t[:, :, i].float() / vmax
        scale = (rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255
        offset = rgb_map[channel]['range'][0] / 255
        x = ch / scale + offset
        x = torch.clamp(x, max=1.0)
        rgb = rgb_map[channel]['rgb'].float()
        x_rgb = (x.unsqueeze(-1) * rgb).round().to(torch.int32)
        colored.append(x_rgb)
    im = torch.stack(colored).sum(dim=0)
    im = torch.clamp(im, max=255).to(torch.uint8)
    return im


def image_path(dataset: str,
               experiment: str,
               plate: int,
               address: str,
               site: int,
               channel: int,
               base_path: str = DEFAULT_IMAGES_BASE_PATH) -> str:
    """Construct the file path of a single channel image."""
    return os.path.join(base_path, dataset, experiment,
                        f"Plate{plate}", f"{address}_s{site}_w{channel}.png")


def load_site(dataset: str,
              experiment: str,
              plate: int,
              well: str,
              site: int,
              channels: Tuple[int, ...] = DEFAULT_CHANNELS,
              base_path: str = DEFAULT_IMAGES_BASE_PATH) -> torch.Tensor:
    paths = [image_path(dataset, experiment, plate, well, site, c, base_path) for c in channels]
    return load_images_as_tensor(paths)


def load_site_as_rgb(dataset: str,
                     experiment: str,
                     plate: int,
                     well: str,
                     site: int,
                     channels: Tuple[int, ...] = DEFAULT_CHANNELS,
                     base_path: str = DEFAULT_IMAGES_BASE_PATH,
                     rgb_map: Dict[int, Dict[str, torch.Tensor]] = RGB_MAP) -> torch.Tensor:
    x = load_site(dataset, experiment, plate, well, site, channels, base_path)
    return convert_tensor_to_rgb(x, channels, rgb_map=rgb_map)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_dataset(base_path: str, dataset: str, include_controls: bool = True) -> pd.DataFrame:
    df = _read_csv(os.path.join(base_path, dataset + '.csv'))
    if include_controls:
        controls = _read_csv(os.path.join(base_path, dataset + '_controls.csv'))
        df['well_type'] = 'treatment'
        df = pd.concat([controls, df], sort=True)
    df['cell_type'] = df.experiment.str.split('-').apply(lambda a: a[0])
    df['dataset'] = dataset
    dfs = []
    for site in (1, 2):
        tmp = df.copy()
        tmp['site'] = site
        dfs.append(tmp)
    res = pd.concat(dfs).sort_values(by=['id_code', 'site']).set_index('id_code')
    return res


def combine_metadata(base_path: str = DEFAULT_METADATA_BASE_PATH,
                     include_controls: bool = True) -> pd.DataFrame:
    df = pd.concat([
        _load_dataset(base_path, dataset, include_controls=include_controls)
        for dataset in ['test', 'train']
    ], sort=True)
    return df


if __name__ == '__main__':
    base_dir ="D:\\PycharmProject\\重要数据\\rxrx1-images\\rxrx1\\HUVEC"
    for subfolder in os.listdir(base_dir):
        path = os.path.join(base_dir, subfolder)
        for ssubfoldr in os.listdir(path):
            path1 = os.path.join(path, ssubfoldr)
            print(path1)
            if "RGB" in os.path.basename(path1):
                continue
            i=1
            paths = []
            for pic in os.listdir(path1):
                paths.append(os.path.join(path1, pic))
                if(i == 6):
                  base = pic.rsplit('_', 1)[0]
                  x = load_images_as_tensor(paths)
                  parent_dir = os.path.dirname(path1)
                  name = os.path.basename(path1)
                  rgb_dir = os.path.join(parent_dir, f"{name}-RGB")
                  if not os.path.isdir(rgb_dir):
                    os.makedirs(rgb_dir, exist_ok=True)
                  im = convert_tensor_to_rgb(x, rgb_map=RGB_MAP)
                  im_np = im.cpu().numpy()  # -> ndarray, dtype uint8
                  # 2. 用 PIL 保存
                  img = Image.fromarray(im_np, mode="RGB")
                  save_path = os.path.join(rgb_dir, base + ".png")
                  print(save_path)
                  img.save(save_path)
                  i = 0
                  paths = []
                i += 1




