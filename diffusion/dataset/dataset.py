from abc import ABC
import os
import sys
from pathlib import Path

import torchvision

import PIL
from PIL import Image
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, train=True, transform=torchvision.transforms.ToTensor(), target_transform=None):
        super().__init__(root=ROOT_DIR,
                         train=train,
                         transform=transform,
                         download=True,
                         target_transform=target_transform)

def transform_Flowers102():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((96, 128)),
        torchvision.transforms.ToTensor(),
        ])

class Flowers102(torchvision.datasets.Flowers102):
    def __init__(self, train=True, transform=transform_Flowers102(), target_transform=None):
        super().__init__(root=ROOT_DIR,
                         split='train' if train else 'val',
                         transform=transform,
                         download=True,
                         target_transform=target_transform)


