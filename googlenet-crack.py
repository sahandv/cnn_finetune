# -*- coding: utf-8 -*-
import os; os.environ['KERAS_BACKEND'] = 'theano'

from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, concatenate
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.models import Model

from sklearn.metrics import log_loss

from custom_layers.googlenet_custom_layers import LRN, PoolHelper

import keras
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from custom_layers.scale_layer import Scale

import pandas as pd

from subprocess import check_output
K.set_image_dim_ordering('th')
