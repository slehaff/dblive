# import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import PIL
from PIL import Image, ImageOps
import random

inputFolder = '/home/samir/Desktop/blender/pycode/15may21/batchsc/'
outputFolder = '/home/samir/Desktop/blender/pycode/15may21/batchsc/'

def reducedcopy(input,output):
    IMAGECOUNT = len(os.listdir(inputFolder))
    for i in range(IMAGECOUNT):
        folder = inputFolder + 'render'+str(i)+'/'
        os.mkdir(outputFolder+'render'+str(i))
        # os.mkdir(inputFolder+'render'+str(i+2*IMAGECOUNT))
        
        image0= cv2.imread(folder+'image0.png')
        image8= cv2.imread(folder+'image8.png')
        image9= cv2.imread(folder+'image9.png')
        im_wrap1= cv2.imread(folder+'im_wrap1.png')
        kdata= cv2.imread(folder+'kdata.png')

        cv2.imwrite(outputFolder+'render'+str(i)+'/' +'image0.png', image0)
        cv2.imwrite(outputFolder+'render'+str(i)+'/' +'image8.png', image8)
        cv2.imwrite(outputFolder+'render'+str(i)+'/' +'image9.png', image9)
        cv2.imwrite(outputFolder+'render'+str(i)+'/' +'im_wrap1.png', im_wrap1)
        cv2.imwrite(outputFolder+'render'+str(i)+'/' +'kdata.png', kdata)



def scalecrop(input, l, offset):
    resized = cv2.resize(input, (l,l), cv2.INTER_AREA)
    print(l, offset)
    cropped = resized[offset:offset+160, offset:offset+160]
    return(cropped)


def resized(input, output, count0, min, max):
    IMAGECOUNT = len(os.listdir(input))
    for i in range(IMAGECOUNT):
        folder = input + 'render'+str(i)+'/'
        os.mkdir(output+'render'+str(i+count0))

        l = random.randint(min,max)
        offset = random.randint(0, l-160)

        image0= cv2.imread(folder+'image0.png')
        image8= cv2.imread(folder+'image8.png')
        image9= cv2.imread(folder+'image9.png')
        im_wrap1= cv2.imread(folder+'im_wrap1.png')
        kdata= cv2.imread(folder+'kdata.png')

        image0= scalecrop(image0, l, offset)
        image8= scalecrop(image8, l, offset)
        image9= scalecrop(image9, l, offset)
        im_wrap1= scalecrop(im_wrap1, l, offset)
        kdata= scalecrop(kdata, l, offset)


        cv2.imwrite(output+'render'+str(i+count0)+'/' +'image0.png', image0)
        cv2.imwrite(output+'render'+str(i+count0)+'/' +'image8.png', image8)
        cv2.imwrite(output+'render'+str(i+count0)+'/' +'image9.png', image9)
        cv2.imwrite(output+'render'+str(i+count0)+'/' +'im_wrap1.png', im_wrap1)
        cv2.imwrite(output+'render'+str(i+count0)+'/' +'kdata.png', kdata)


resized(inputFolder, outputFolder,2000,160,270)


