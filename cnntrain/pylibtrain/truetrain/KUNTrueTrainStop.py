# changed sunday 11: Autoencoder module for instant depth determination
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import pandas as pd


import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import kerastuner as kt
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add
from tensorflow.keras.layers import Input, Activation, UpSampling2D, add
from tensorflow.keras.utils import plot_model
from datetime import datetime
from packaging import version


########################################################################### tf bug fix    ############################################################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#################################################################################################################################################################



H = 160
W = 160

EPOCHS = 100
inputFolder = '/home/samir/Desktop/blender/pycode/15may21/batch/'
IMAGECOUNT = 1000# len(os.listdir(inputFolder))-2


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
        myfile = folder_path + str(i)+'/'+ filename + '.png'
        img = cv2.imread(myfile).astype(np.float32)
        img = resize(img, 160, 160)
        print('img:', img.shape)
        img = normalize_image255(img)
        inp_img = make_grayscale(img)
        array.append(inp_img)
    return   


def to_new_npy_array(folder_path, filename, array, file_count):
    for i in range(0, file_count, 1):
        myfile = folder_path + str(i)+str(i)+'/'+ filename +'.npy'
        img = np.load(myfile)
        img = normalize_image255(img)
        array.append(img)
        print('npyarray shape:', np.shape(array))
    return    


def to_npy_array(folder_path, array, file_count):
    for i in range(0, file_count, 1):
        myfile = folder_path + str(i)+'.npy'
        img = np.load(myfile)
        img = normalize_image255(img)
        array.append(img)
        print('npyarray shape:', np.shape(array))
    return


# Load and pre-process the training data
wrap_images = []
k_images = []
#========================================= Use with dblive folder structure ===============================
to_png_array(inputFolder+'render', 'im_wrap1', wrap_images, IMAGECOUNT)
to_png_array(inputFolder+'render', 'kdata' , k_images, IMAGECOUNT)


#========================================= Use with serverless folder structure ===============================
# to_array('/home/samir/serverless/new1-469/1/wrap/', fringe_images, IMAGECOUNT)
# to_array('/home/samir/serverless/new1-469/unwrap/', unwrap_images, IMAGECOUNT)

# Expand the image dimension to conform with the shape required by keras and tensorflow, inputshape=(..., h, w, nchannels).
wrap_images = np.expand_dims(wrap_images, -1)
k_images = np.expand_dims(k_images, -1)
print("input shape: {}".format(wrap_images.shape))
# print("output shape: {}".format(nom_images.shape))
print(len(wrap_images))




inputImage = Input(shape=(H,W,1))

A1 = Conv2D(32,(3,3), padding='same')(inputImage)
A2 = Activation('relu')(A1)
A3 = Conv2D(32,(3,3), padding='same')(A2)
A4 = Activation('relu')(A3)
print(A4.shape)

A5 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same')(A4)
A6 = Conv2D(64,(3,3), padding='same')(A5)
A7 = Activation('relu')(A6)
A8 = Conv2D(64,(3,3), padding='same')(A7)
A9 = Activation('relu')(A8)
print(A9.shape)

A10 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same')(A9)
A11 = Conv2D(128,(3,3), padding='same')(A10)
A12 = Activation('relu')(A11)
A13 = Conv2D(128,(3,3), padding='same')(A12)
A14 = Activation('relu')(A13)
print(A14.shape)

A15 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same')(A14)
A16 = Conv2D(256,(3,3), padding='same')(A15)
A17 = Activation('relu')(A16)
A18 = Conv2D(256,(3,3), padding='same')(A17)
A19 = Activation('relu')(A18)
print(A19.shape)

A20 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same')(A19)
A21 = Conv2D(512,(3,3), padding='same')(A20)
A22 = Activation('relu')(A21)
A23 = Conv2D(512,(3,3), padding='same')(A22)
A24 = Activation('relu')(A23)
print(A24.shape)

A25 = MaxPooling2D(pool_size=(2,2), strides=None, padding='same')(A24)
A26 = Conv2D(1024,(3,3), padding='same')(A25)
A27 = Activation('relu')(A26)
A28 = Conv2D(1024,(3,3), padding='same')(A27)
A29 = Activation('relu')(A28)
print(A29.shape)

############################################# Decode ###############################################

A30 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A29)
A31 = Conv2D(512,(3,3), padding='same')(A30)
A32 = Activation('relu')(A31)
A33 = add([A24, A32])
A34 = Conv2D(512,(3,3), padding='same')(A33)
A35 = Activation('relu')(A34)
print(A35.shape)

A36 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A35)
A37 = Conv2D(256,(3,3), padding='same')(A36)
A38 = Activation('relu')(A37)
A39 = add([A19, A38])
A40 = Conv2D(256,(3,3), padding='same')(A39)
A41 = Activation('relu')(A40)
print(A41.shape)

A42 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A41)
A43 = Conv2D(128,(3,3), padding='same')(A42)
A44 = Activation('relu')(A43)
A45 = add([A14, A44])
A46 = Conv2D(128,(3,3), padding='same')(A45)
A47 = Activation('relu')(A46)
print(A47.shape)

A48 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A47)
A49 = Conv2D(64,(3,3), padding='same')(A48)
A50 = Activation('relu')(A49)
A51 = add([A9, A50])
A52 = Conv2D(64,(3,3), padding='same')(A51)
A53 = Activation('relu')(A52)
print(A53.shape)

A54 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A53)
A55 = Conv2D(32,(3,3), padding='same')(A54)
A56 = Activation('relu')(A55)
A57 = add([A4, A56])
A58 = Conv2D(32,(3,3), padding='same')(A57)
A59 = Activation('relu')(A58)
print(A59.shape)
# Output Conv2D(1)
outputImage = Conv2D(1, (3, 3), padding='same')(A59)


UModel = Model(inputImage, outputImage)
UModel.summary()



def compile_model(model):
    # model = Model(input_image, output_image)
    sgd = optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mse'])
    model.summary()
    return(model)



loss = []
val_loss = []
convweights = []

compile_model(UModel)
model = UModel


def load_model():
    model = tf.keras.models.load_model(
        '/home/samir/dblive/cnnpredict/models/UN15models/UN15may-batch3K-Kunw-b8-100.h5')
    model.summary()
    return(model)


model = load_model()


# tf.keras.utils.plot_model(
#    model, to_file='model.png', show_shapes=False, show_dtype=False,
#     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
# )

checkpointer = ModelCheckpoint(
    filepath="weights/weights.hdf5", verbose=1, save_best_only=True)


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=3,verbose=1, restore_best_weights=True)




history = model.fit(wrap_images, k_images,
                            batch_size=2,
                            epochs=EPOCHS,
                            validation_split=0.2,
                            callbacks=[tensorboard_callback, checkpointer])#, early_callback])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



def plot():
    # Plot the training and validation losses
    plt.close('all')

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.show()
    plt.savefig('trainingvalidationlossgx.png')


# plot()


def combImages(i1, i2, i3):
    new_img = np.concatenate((i1, i2, i3), axis=1)
    return(new_img)


def DB_predict(i, x, y):
    predicted_img = model.predict(np.array([np.expand_dims(x, -1)]))
    predicted_img = predicted_img.squeeze()
    # cv2.imwrite('validate/'+str(i)+'filteredSync.png',
    #             (255.0*predicted_img).astype(np.uint8))
    # cv2.imwrite('validate/'+str(i)+'input.png',
    #             (255.0*x).astype(np.uint8))
    combo = combImages(255*x,3*(np.round((255*predicted_img)/3)), 255*y)
    cv2.imwrite('validate/test/'+str(i)+'combo.png', (1.0*combo).astype(np.uint8))
    return(combo)


# get_my_file('inp/' + str(1)+'.png')
myfile = inputFolder+'render' + str(1)+'/im_wrap1.png'
img = cv2.imread(myfile).astype(np.float32)
img = resize(img, 160, 160)
img = normalize_image255(img)
inp_img =  make_grayscale(img)
combotot = combImages(inp_img, inp_img, inp_img)
for i in range(0, 90, 1):
    print(i)
    # get_my_file('inp/' + str(i)+'.png')
    myfile = inputFolder+'render' + str(i)+'/im_wrap1.png'
    print(myfile)
    img = cv2.imread(myfile).astype(np.float32)
    img = resize(img, 160, 160)
    img = normalize_image255(img)
    inp_img = make_grayscale(img)
    #get_my_file('out/' + str(i)+'.png')
    myfile = inputFolder+'render' + str(i)+'/kdata.png'
    img = cv2.imread(myfile).astype(np.float32)
    img = resize(img, 160, 160)
    img = normalize_image255(img)
    out_img = make_grayscale(img)
    # out_img = np.round(out_img/2)
    combo = DB_predict(i, inp_img, out_img)
    combotot = np.concatenate((combotot, combo), axis=0)
model.save('/home/samir/dblive/cnnpredict/models/UN15models/UN15may-batch3K-Kunw-b8-100.h5', save_format='h5')
cv2.imwrite('validate/'+'UN15may-44x-Ntarget-Kunw-b8-20.png',
            (1.0*combotot).astype(np.uint8))
