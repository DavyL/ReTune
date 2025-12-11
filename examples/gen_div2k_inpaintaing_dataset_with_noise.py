import deepinv as dinv
import torch
import os
from pathlib import Path
from torchvision import transforms

BASE_DIR = Path(".")
print(BASE_DIR)
MEASUREMENT_DIR = BASE_DIR / "measurements"
DATA_DIR = BASE_DIR / "datasets"
CKPT_DIR = BASE_DIR / "ckpts"

dataset_name = 'div2k'
#operation = 'inpainting_with_noise'
operation = 'inpainting_01'
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
    download=False,
)

test_dataset = dinv.datasets.DIV2K(
    root=DATA_DIR,
    mode="val",
    transform=test_transform,
    download=False,
)

n_channels = 3  # 3 for color images, 1 for gray-scale images
#probability_mask = 0.5  # probability to mask pixel
probability_mask = 0.1  # probability to mask pixel

#physics = dinv.physics.Denoising( device=device)
physics = dinv.physics.Inpainting(
    tensor_size=(n_channels, img_size, img_size), mask=probability_mask, device="cpu", noise_model=dinv.physics.GaussianNoise()
)
#physics = dinv.physics.Blur(
#    dinv.physics.blur.gaussian_blur(sigma=(.5, 0.5), angle=0), device=device, padding='reflect'
#)

#dataset_root = os.path.dirname(measurement_dir)
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

#trainval_dataset = dinv.datasets.HDF5Dataset(dataset_path, train=True)