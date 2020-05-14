
import time
import os
import cv2
import numpy as np

SAMPLING_RATE = 100
SAVE_RATE = 1

vidcap = cv2.VideoCapture('data/train.mp4')
labels_file = 'data/train.txt'
training_data = []


def getSpeedList(file):
    with open(file) as f:
        lines = [line.rstrip('\n') for line in f]

    return lines


labels = getSpeedList(labels_file)


starting_value = 1
file_name = 'multiset/training_data-{}.npy'.format(starting_value)
success, image = vidcap.read()
count = 0
training_data = []

while True:
    if len(training_data) % 100 == 0:
        print(len(training_data))

    count += 1
    speed = labels[count]
    training_data.append([image, speed])
    success, image = vidcap.read()

    # Save a checkpoint
    if len(training_data) == 20399:
        np.save(file_name, training_data)
        training_data = []
        starting_value += 1
        file_name = 'training_data-{}.npy'.format(starting_value)
        print("SAVED TRAINING DATA", file_name)
