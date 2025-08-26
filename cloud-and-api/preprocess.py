import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torchvision.datasets import ImageFolder
import cv2
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from google.cloud import storage

# data = tf.keras.preprocessing.image_dataset_from_directory(
#     '/content/drive/MyDrive/selen_imaging/mini_bucket',  # name of the file where they all exist
#     labels='inferred',                        # assigns labels based on the subfolder names
#     label_mode='int',                         # format of the labels
#     class_names=None,                         # order of class names, none is the default that does it alphabetically
#     color_mode='rgb',                         # determines the channels they are loaded with, (greyscale possible)
#     batch_size=64,                            # Number of images in each batch.
#     image_size=(1024, 1024),                  # determine the size of the images you want them in
#     shuffle=True,                             # randomly mix up the images to avoid bias if they are organized
#     seed=None,                                # random seed or label for shuffling or splitting
#     # validation_split=None,                  # used for validation set
#     # subset=None,                            # specify if you want the training or validation split
#     interpolation='bilinear',                 # how to resize the images (teres other options)
#     follow_links=False,                       # to not follow symbolic links inside the dataset directory
#     crop_to_aspect_ratio=False,               # not to crop images to keep their aspect ratio
#     pad_to_aspect_ratio=False,                # to maintain the aspect ratio
#     data_format=None,                         # format will be (batch, height, width, channels) which is the default
#     verbose=True                              # prints the progress message while loading
# )
