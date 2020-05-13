
import os
import sys
import cv2
import numpy as np 
import pandas as pd 
from collections import Counter 

# arg parse for wether or not we want to view data
VIEW = False
SAMPLE_NUMBER = 1
try:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    print(arg1)
    print(arg2)
    if arg1 == 'v' or arg1 == 'V':
        VIEW = True
        SAMPLE_NUMBER = arg2
except:
    print("Try running again with 'v' flag to view\npython balance_data.py v")

dataset = os. listdir('data/')

def view_dataset(dataset, view=False):
    if view: 
        print(f'Viewing data/training_data-{SAMPLE_NUMBER}.npy')
        train_data = np.load(f'data/training_data-{SAMPLE_NUMBER}.npy', allow_pickle=True)
        for data in train_data:
            img = data[0]
            choice = data[1]

            cv2.imshow('test', img)
            print(f'Speed: {choice}\nShape: {img.shape}')

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break    

view_dataset(dataset, VIEW)
