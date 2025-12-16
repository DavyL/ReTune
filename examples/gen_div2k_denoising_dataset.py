#%%

import deepinv as dinv
import torch
import os
from pathlib import Path
from torchvision import transforms
from retune.ChannelNoise import ChannelNoiseModel
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
print(f'device is {device}')

BASE_DIR = Path(".")
print(BASE_DIR)
MEASUREMENT_DIR = BASE_DIR / "measurements"
DATA_DIR = BASE_DIR / "datasets"
CKPT_DIR = BASE_DIR / "ckpts"

dataset_name = 'div2k'
operation = 'denoising_sigma-aniso'
measurement_dir = MEASUREMENT_DIR / dataset_name / operation 
img_size = 256 

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.CenterCrop(img_size)]
)
train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomCrop(img_size)]
)


train_dataset = dinv.datasets.DIV2K(
    root=DATA_DIR,
    mode="train",
    transform=train_transform,
    download=True,
)

test_dataset = dinv.datasets.DIV2K(
    root=DATA_DIR,
    mode="val",
    transform=test_transform,
    download=True,
)
noise_model = ChannelNoiseModel(sigmas=[0.1,0.25,0.5])
physics = dinv.physics.Denoising(noise_model)

os.makedirs(measurement_dir, exist_ok=True)
dinv_dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    save_dir=measurement_dir,
    batch_size=32,
    device="cpu",
)
print(f'dataset was generated at  {dinv_dataset_path}')

# %%
