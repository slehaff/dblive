# cannot easily visualize filters lower down
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot

# plot feature map of first conv layer for given image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math
import os.path
from os import path
import tensorflow as tf
# add to the top of your code under import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.python.keras.layers import Layer, InputSpec
# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Lambda, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, UpSampling2D, concatenate, Concatenate, AveragePooling2D
from tensorflow.keras.utils import plot_model
import time
# import nnwrap as nnw
import argparse
import sys
# import myglobals




def load_H_model():
    # model = tensorflow.keras.models.load_model('/home/samir/dblive/cnnpredict/models/UNmodels/UNet02-224-fringe-wrapdata'+'-200-adam-noBN.h5')
    model = tensorflow.keras.models.load_model('/home/samir/dblive/cnnpredict/models/UN15models/UN15-551-tMan-b8-Wrap-200-V2.h5')
    return(model)
# /home/samir/dblive/cnnpredict/models/cnnres01-220-modelwrap1'+'-200-adam-noBN.h5

def load_L_model():
    model = tensorflow.keras.models.load_model('/home/samir/dblive/cnnpredict/models/UN15models/UN15-551-tMan-b8-KUnw-300-V2.h5')
    return(model)


# load the model
model = VGG16()
# model = load_H_model()
# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()


