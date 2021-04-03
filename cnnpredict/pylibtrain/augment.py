# import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import PIL
from PIL import Image, ImageOps

inputFolder = '/home/samir/Desktop/blender/pycode/30train2T/'
oufolder = '/home/samir/Desktop/blender/pycode/30train2TM/'
IMAGECOUNT = len(os.listdir(inputFolder))
for i in range(IMAGECOUNT):
    folder = inputFolder + 'render'+str(i)+'/'
    for j in range(10):
        im = Image.open(folder+'image'+str(j)+'.png')
        mirror = ImageOps.mirror(im)
        # image= cv2.imread(folder+'image'+str(j)+'.png')
        # flipped = cv2.flip(image,1)
        mirror.save('/home/samir/Desktop/blender/pycode/30train2TM/'+'render'+str(i)+'/' +'image'+str(j)+'.png')
        print('/home/samir/Desktop/blender/pycode/30train2TM/'+'render'+str(i)+'/' +'image'+str(j)+'.png')