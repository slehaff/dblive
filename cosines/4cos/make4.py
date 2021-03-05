import cv2
import numpy as np
from PIL import Image

width = 4096
height = 2732
hfperiods = 160
periods = 6


def maketexture(h, w, value):
    ima = np.full((w,h), value)
    print('texture')
    # ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/4cos/' + 'texture.png', ima)


def makeblack(h, w, value):
    ima = np.full((w,h), value)
    # ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/4cos/' + 'black.png', ima)




def makeimage(h, w, wvcount, phi):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    for i in range(w):
        imaline[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(float(phi)/4.0 + wvcount*float(i)/float(w))))
    print(imaline)
    for j in range(h):
        ima[:, j] = imaline
    # ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/4cos/'+ str(phi + 1) + '_cos.jpg', ima)

# folder = "/home/samir/dblive/cosines/singleshotcosines/"

makeimage(width, height, hfperiods, -1)
makeimage(width, height, hfperiods, 0)
makeimage(width, height, hfperiods, 1)
makeimage(width, height, hfperiods, 2)
maketexture(width,height,200)
makeblack(width,height,2)
makeimage(width, height, periods, 5)
makeimage(width, height, periods, 6)
makeimage(width, height, periods, 7)
makeimage(width, height, periods, 8)