from tensorflow.keras import layers
import numpy as np
import os
import cv2
import pandas as pd
import keras_preprocessing
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, concatenate
from tensorflow.keras.layers import Input, Activation, UpSampling2D, add
from tensorflow.keras.metrics import MeanIoU
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import glob


H = 160
W = 160
N = 18 # Number of classes corresponds to hfwrap periods

EPOCHS = 40
inputFolder = '/home/samir/Desktop/blender/pycode/15may21/segbatch/'
IMAGECOUNT = 3000 #len(os.listdir(inputFolder))
n_classes = 18 # Still number of classes

def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def resize(img, w,h):
    print(img.shape)
    img = img[0:160,0:160]
    print(img.shape)
    return(img)


def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)number_of_epochs = 5
    img = img/255.0
    return img


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    print(np.max(img), np.min(img))
    img = img - np.min(img)
    img = img/np.max(img)
    return img


def to_array(folder_path, array, file_count):
    for i in range(0, file_count, 1):
        myfile = folder_path + str(i)+'.png'
        img = cv2.imread(myfile).astype(np.float32)
        img = resize(img, 160, 160)
        print('img:', img.shape)
        img = normalize_image255(img)
        inp_img = make_grayscale(img)
        array.append(inp_img)
    return


def to_png_array(folder_path, filename, array, file_count):
    for i in range(file_count):
        print('count:', i)
        myfile = folder_path + filename +str(i)+ '.png'
        img = cv2.imread(myfile).astype(np.float32)
        img = resize(img, 160, 160)
        print('img:', img.shape)
        img = normalize_image255(img)
        inp_img = make_grayscale(img)
        array.append(inp_img)
    return   


def to_new_npy_array(folder_path, filename, array, file_count):
    for i in range(0, file_count, 1):
        myfile = folder_path+filename +str(i)+'.npy'
        img = np.load(myfile)
        # img = normalize_image255(img)
        array.append(img)
        print('npyarray shape:', np.shape(array))
    return    


def to_npy_array(folder_path, array, file_count):
    for i in range(0, file_count, 1):
        myfile = folder_path + str(i)+'.npy'
        num = np.load(myfile)
        img = normalize_image255(img)
        array.append(img)
        print('npyarray shape:', np.shape(array))
    return



# Load and pre-process the training data
wrap_images = []
k_masks = []
#========================================= Use with dblive folder structure ===============================
to_png_array(inputFolder+ 'wrap/', 'img',wrap_images, IMAGECOUNT)
to_png_array(inputFolder+ 'k/','img' , k_masks, IMAGECOUNT)

wrap_images = np.array(wrap_images)
k_masks = np.array(k_masks)

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = k_masks.shape
print(k_masks.shape)
train_masks_reshaped = k_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(np.ravel(train_masks_reshaped))
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(wrap_images, train_masks_input, test_size = 0.10, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.5, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))