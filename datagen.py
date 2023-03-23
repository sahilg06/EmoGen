import os
from os.path import dirname, join, basename, isfile, isdir, splitext
import numpy as np
import random 
from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset
import cv2

from hparams import hparams

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

emotion_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
intensity_dict = {'XX':0, 'LO':1, 'MD':2, 'HI':3}
emonet_T = 5

class Dataset(object):
    def __init__(self, args, val=False):
        self.args = args
        self.filelist = []

        if not val:
            self.path = self.args.in_path
        else:
            self.path = self.args.val_path
        
        self.all_videos = [f for f in os.listdir(self.path) if isdir(join(self.path, f))]

        for filename in self.all_videos:
            #print(splitext(filename))
            labels = splitext(filename)[0].split('_')
            emotion = emotion_dict[labels[2]]
            
            emotion_intensity = intensity_dict[labels[3]]
            if val:
                if emotion_intensity != 3:
                    continue
            
            self.filelist.append((filename, emotion, emotion_intensity))

        self.filelist = np.array(self.filelist)
        print('Num files: ', len(self.filelist))

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + emonet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.filelist) - 1)
            filename = self.filelist[idx]
            vidname = filename[0]
            emotion = int(filename[1])
            emotion = to_categorical(emotion, num_classes=6)
            emotion_intensity = int(filename[2]) # We don't use this info

            img_names = list(glob(join(self.path, vidname, '*.jpg')))

            if len(img_names) <= 3 * emonet_T:
                continue
            img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)

                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break
                window.append(img)

            if not all_read: continue

            # H x W x 3 * T
            # x = np.concatenate(window, axis=2) / 255.
            x = np.asarray(window) / 255
            x = x.transpose(3, 0, 1, 2) # C, T, W, H

            x = torch.FloatTensor(x)
            # print(x.shape)

            return x, emotion

