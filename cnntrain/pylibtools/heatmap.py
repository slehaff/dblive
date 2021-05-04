import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import cv2
import os


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


bfolder = '/home/samir/Desktop/blender/pycode/15trainMan/'

for i in range(0,len(os.listdir(bfolder)), 40):
    print(i)
    folder = bfolder +'render'+ str(i)
    print(folder)
    f1 = folder + '/depth.png'
    imganal = cv2.imread(f1, 1)
    imganal = make_grayscale(imganal)

    f2 = folder + '/nndepth.png'
    imgnn = cv2.imread(f2, 1)
    imgnn = make_grayscale(imgnn)

    dif = np.abs(imgnn-imganal)
    heatmap = sb.heatmap(dif)
    plt.show()
