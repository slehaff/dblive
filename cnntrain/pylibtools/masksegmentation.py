

import os
import tensorflow as tf
import keras
SM_FRAMEWORK=tf.keras
# import segmentation_models as sm
# from segmentation_models import Unet
# from segmentation_models import get_preprocessing
# from segmentation_models.losses import bce_jaccard_loss
# from segmentation_models.metrics import iou_score
import cv2
import shutil
import numpy as np
classes = 18
H = 160
W = 160

# cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)



# model = Unet()
infolder = '/home/samir/Desktop/blender/pycode/15may21/testbatch/'
wrapfolder ='/home/samir/Desktop/blender/pycode/15may21/segbatch/wrap/'
kfolder ='/home/samir/Desktop/blender/pycode/15may21/segbatch/k/'
mfolder ='/home/samir/Desktop/blender/pycode/15may21/segbatch/'
hotkfolder = '/home/samir/Desktop/blender/pycode/15may21/segbatch/hotkey/'
nnkfolder = '/home/samir/Desktop/blender/pycode/15may21/segbatch/nnk/'
######################################## Data Prepare ################################
def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img



def mask(folder):
    color = folder + 'blendertexture.png'
    print('color:', color)
    img1 = np.zeros((H, W), dtype=np.float)
    img1 = cv2.imread(color, 1).astype(np.float32)
    gray = make_grayscale(img1)
    black = folder + 'blenderblack.png'
    img2 = np.zeros((H, W), dtype=np.float)
    img2 = cv2.imread(black, 0).astype(np.float32)
    diff1 = np.subtract(gray, .5*img2)
    mask =  np.zeros((H, W), dtype=np.float)
    for i in range(H):
        for j in range(W):
            if (diff1[i,j]<50):
                mask[i,j]= True
    np.save( folder+ 'mask.npy', mask, allow_pickle=False)
    cv2.imwrite( folder+ 'mask.png', 128*mask)
    return(mask)


def applymask(inputfolder,inputfile):
    img = np.zeros((H, W), dtype=np.float)
    img = cv2.imread(inputfolder+inputfile, 0).astype(np.float32)
    mask = np.load(inputfolder+'/mask.npy')
    masked = np.multiply(np.logical_not(mask), img)
    return(masked)


def reshuffledata():
    for i in range(len(os.listdir(infolder))):
        print(i)
        masked =applymask(infolder + 'render'+ str(i) , '/nnkdata.png')
        masked = np.round(masked*17/(np.max(masked)))
        # print( 'max:', np.max(masked))
        cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(nnkfolder + 'img'+ str(i) +  '.png', masked)   

def shuffledata():
    for i in range(len(os.listdir(infolder))):
        masked =applymask(infolder + 'render'+ str(i) , '/kdata.png')
        masked = np.round(masked*17/(np.max(masked)))
        # print( 'max:', np.max(masked))
        cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(kfolder + 'img'+ str(i) +  '.png', masked)
        hotmask = np.zeros((H,W,classes))
        for c in range(classes):
            cmask = np.zeros((H, W), np.bool)
            cmask = np.logical_not(np.abs(c-masked)) #np.logical_not
            # print (cmask)
            print(mfolder+str(c)+'mask/'+'img'+str(i)+'.png')
            cv2.imwrite(mfolder+str(c)+'mask/'+'img'+str(i)+'.png', cmask*1 )
            print('c:',c)
            hotmask[:,:,c] = cmask
        np.save(hotkfolder+ 'npimg'+ str(i)+'.npy', hotmask)
        masked =applymask(infolder + 'render'+ str(i) , '/im_wrap1.png')
        cv2.imwrite(wrapfolder + 'img'+ str(i) +  '.png', masked)
        print(i)

shuffledata()
# reshuffledata()
