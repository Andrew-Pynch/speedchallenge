from __future__ import print_function, division
import os
import re

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SpeedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory containing images with labels in filename
            transform (callable, optional): Optional transforms to be applied to to a sample
        """
        self.root_dir = root_dir
        self.items = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.items[idx])

        nparr = io.imread(img_name)
        # Convert image to torch tensor
        image = torch.from_numpy(nparr)

        regex = re.split("/|_", img_name)
        label = float(regex[1])
        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def showItem(self, idx):
        image, label = (self[idx]['image'], self[idx]['label'])
        print(label)
        plt.imshow(image)


"""
data = SpeedDataset(root_dir='vidCaps/')

sample = data[0]
print(sample['label'], 'MPH')
plt.imshow(sample['image'])
"""
