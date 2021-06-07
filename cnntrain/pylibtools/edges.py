import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

bfolder = '/home/samir/Desktop/blender/pycode/15may21/batchsc/'

for i in range(20):
    img1 = cv.imread(bfolder+'render'+str(i) + '/kdata.png',0)
    edges1 = cv.Canny(img1,15,35)
    img2 = cv.imread(bfolder+'render'+str(i) +  '/nnkdata.png',0)
    edges2 = cv.Canny(img2,10,35)

    plt.subplot(121),plt.imshow(edges1,cmap = 'gray')
    plt.title('Analytic'+str(i)), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges2,cmap = 'gray')
    plt.title('Neural'+str(i)), plt.xticks([]), plt.yticks([])
    plt.show()