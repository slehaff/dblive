import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

bfolder = '/home/samir/Desktop/blender/pycode/stitch2/render0/'


img1 = cv.imread(bfolder + 'kdata.png',0)
edges1 = cv.Canny(img1,20,35)
img2 = cv.imread(bfolder + 'nnkdata.png',0)
edges2 = cv.Canny(img2,20,35)

plt.subplot(121),plt.imshow(edges1,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges2,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()