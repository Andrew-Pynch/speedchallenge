from __future__ import print_function, division
import os
import re

import cv2
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
        img_name = self.items[idx]
        full_img_name = os.path.join(self.root_dir, img_name)

        nparr = io.imread(full_img_name)
        nparr = cv2.resize(nparr, (480, 480))
        # Convert image to torch tensor
        image = torch.from_numpy(nparr)

        regex = re.split("/|_", full_img_name)
        label = float(regex[1])

        sample = {"fname": img_name, "image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_item(self, idx):
        image, label = (self[idx]["image"], self[idx]["label"])
        plt.title(f"{round(label, 2)}  MPH")

        plt.imshow(image)

    def show_batch(self):
        pass


# data = SpeedDataset(root_dir="vidCaps/")

# image = data[0]["image"]
# print(type(image))
# label = data[0]["label"]
# print(type(label))

# print(data[0]["fname"])
