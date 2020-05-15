import time
import os
import cv2
import numpy as np

vidcap = cv2.VideoCapture("data/train.mp4")
labels_file = "data/train.txt"
training_data = []


def getSpeedList(file):
    with open(file) as f:
        lines = [line.rstrip("\n") for line in f]

    return lines


labels = getSpeedList(labels_file)


success, image = vidcap.read()
count = 0
while success:
    speed = labels[count]
    # cv2.imwrite("jpg_extraction/frame%d.jpg" % count, image)     # save frame as JPEG file
    if count <= 18000:
        cv2.imwrite(f"training_data/{speed}_{count}.jpg", image)
    elif count > 18000:
        cv2.imwrite(f"validation_data/{speed}_{count}.jpg", image)
    success, image = vidcap.read()
    print(f"Read {count}th new frame: ", success)
    count += 1
