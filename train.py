import os
import glob
from random import shuffle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras import optimizers
from keras import backend as K
# from utils import Sample

FILE_I_END = 1

EPOCHS = 3
BATCH_SIZE = 10

WIDTH = 480
HEIGHT = 640
DIMENSION = 3

INPUT_SHAPE = (WIDTH, HEIGHT, DIMENSION)
MODEL_NAME = 'test_model_v2.69'

OUT_SHAPE = 1 # Vector of len 10 where each element is a value for a control output

def customized_loss(y_true, y_pred, loss='euclidean'):
    # Simply a mean squared error that penalizes large joystick summed values
    if loss == 'L2':
        L2_norm_cost = 0.001
        val = K.mean(K.square((y_pred - y_true)), axis=-1) \
                    + K.sum(K.square(y_pred), axis=-1)/2 * L2_norm_cost
    # euclidean distance loss
    elif loss == 'euclidean':
        val = K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))
    return val


def create_model(keep_prob = 0.8):
    model = Sequential()

    # NVIDIA's model
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=INPUT_SHAPE))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation='softsign'))
    return model


# Handy function to properly sort files
def order_files_by_date(path_to_folder, file_type):
    files = glob.glob("%s*%s" % (path_to_folder, file_type))
    files.sort(key=os.path.getmtime)
    return files
    

data_files = order_files_by_date('dataset/', '.npy')
model = create_model()
model.compile(loss=customized_loss, optimizer=optimizers.adam())



print(len(data_files))

for i, file in enumerate(data_files):
    # Load Training Data
    data = np.load(file, allow_pickle=True)
    # print(data.shape)
    # print(data[0][0].shape)
    # print(type(data[0][1]))
    X = np.array([i[0] for i in data]).reshape(-1,WIDTH,HEIGHT,3)
    y = [float(data[0][1])]
    print(i)

    model.fit(X, y)

model.save_weights(f'{MODEL_NAME}.h5')

