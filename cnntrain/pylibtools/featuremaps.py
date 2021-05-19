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




########################################################################### tf bug fix    ############################################################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#################################################################################################################################################################

def load_H_model():
    # model = tensorflow.keras.models.load_model('/home/samir/dblive/cnnpredict/models/UNmodels/UNet02-224-fringe-wrapdata'+'-200-adam-noBN.h5')
    model = tensorflow.keras.models.load_model('/home/samir/dblive/cnnpredict/models/UN15models/UN15-551-tMan-b8-Wrap-200-V2.h5')
    return(model)

def load_L_model():
    model = tensorflow.keras.models.load_model('/home/samir/dblive/cnnpredict/models/UN15models/UN15may-batch3K-Kunw-b8-100.h5')
    return(model)

def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img

def DB_predict( model, x):
    predicted_img = model.predict(np.array([np.expand_dims(x, -1)]))
    predicted_img = predicted_img.squeeze()
    # tensorflow.keras.backend.clear_session()
    return(predicted_img)




# load the model
# model = VGG16()
model = load_L_model()
# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()
# load the image with the required shape
wrapin ='/home/samir/Desktop/blender/pycode/15trainMan/render0/im_wrap1.png'
# img = load_img(mybird, target_size=(160, 160))
img = cv2.imread(wrapin, 1).astype(np.float32)
img = make_grayscale(img)
# convert the image to an array
# img = img_to_array(img)
# # expand dimensions so that it represents a single 'sample'
# img = expand_dims(img, axis=0)
# # prepare the image (e.g. scale pixel values for the vgg)
# img = preprocess_input(img)
# # get feature map for first hidden layer

img = normalize_image(img)

feature_maps = DB_predict(model, img)
# plot all 64 maps in an 8x8 squares
square = 5
ix = 1
for _ in range(6):
	for _ in range(5):
		# specify subplot and turn of axis
		print('ix =', ix)
		ax = pyplot.subplot(5, 6, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		print("Shape of the array = ",np.shape(feature_maps));
		pyplot.imshow(feature_maps[ :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()