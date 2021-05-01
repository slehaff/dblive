# import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import PIL
from PIL import Image, ImageOps

inputFolder = '/home/samir/Desktop/blender/pycode/15trainMat/'
oufolder = '/home/samir/Desktop/blender/pycode/15trainMat/'
IMAGECOUNT = len(os.listdir(inputFolder))
for i in range(IMAGECOUNT):
    folder = inputFolder + 'render'+str(i)+'/'
    os.mkdir('/home/samir/Desktop/blender/pycode/15trainMat/'+'render'+str(i+IMAGECOUNT))
    for j in range(10):
        im = Image.open(folder+'image'+str(j)+'.png')
        mirror = ImageOps.mirror(im)
        # image= cv2.imread(folder+'image'+str(j)+'.png')
        # flipped = cv2.flip(image,1)
        
        mirror.save('/home/samir/Desktop/blender/pycode/15trainMat/'+'render'+str(i+IMAGECOUNT)+'/' +'image'+str(j)+'.png')
        
        
        print('/home/samir/Desktop/blender/pycode/15trainMat/'+'render'+str(i+IMAGECOUNT)+'/' +'image'+str(j)+'.png')