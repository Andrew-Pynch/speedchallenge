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


# success,image = vidcap.read()
# count = 0
# while success:
#     speed = labels[count]
#     #cv2.imwrite("jpg_extraction/frame%d.jpg" % count, image)     # save frame as JPEG file    
#     cv2.imwrite(f'vidCaps/{speed}_{count}.jpg', image)
#     success,image = vidcap.read()
#     print(f'Read {count}th new frame: ', success)
#     count += 1


starting_value = 1
file_name = 'data/training_data-{}.npy'.format(starting_value)
success, image = vidcap.read()
count = 0
training_data = []
while success:
    count += 1
    speed = labels[count]
    training_data.append([image, speed])
    
    success, image = vidcap.read()

    np.save(file_name, training_data)
    training_data = []
    starting_value += 1
    file_name = 'dataset/training_data-{}.npy'.format(starting_value)
