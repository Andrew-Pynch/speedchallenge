import os
import cv2
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from SpeedDataset import SpeedDataset

seed = np.random.seed(42)

training_data = SpeedDataset("training_data/")
validation_data = SpeedDataset("validation_data/")

training_data[0]["label"]

