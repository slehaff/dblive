# import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import PIL
from PIL import Image, ImageOps

inputFolder = '/home/samir/Desktop/blender/pycode/15may21/batch/'
IMAGECOUNT = len(os.listdir(inputFolder))
for i in range(IMAGECOUNT):
    folder = inputFolder + 'render'+str(i)+'/'
    os.mkdir(inputFolder+'render'+str(i+IMAGECOUNT))
    os.mkdir(inputFolder+'render'+str(i+2*IMAGECOUNT))
    for j in range(10):
        image= cv2.imread(folder+'image'+str(j)+'.png')
        fliph = cv2.flip(image,1)
        flipv = cv2.flip(image,0)
        cv2.imwrite(inputFolder+'render'+str(i+IMAGECOUNT)+'/' +'image'+str(j)+'.png', fliph)
        cv2.imwrite(inputFolder+'render'+str(i+2*IMAGECOUNT)+'/' +'image'+str(j)+'.png', flipv)
        print(inputFolder+'render'+str(i+IMAGECOUNT)+'/' +'image'+str(j)+'.png')