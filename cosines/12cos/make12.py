import cv2
import numpy as np
from PIL import Image

width = 4096
height = 2732
hfperiods = 140
periods = 6


def maketexture(h, w, value):
    ima = np.full((w,h), value)
    print('texture')
    # ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/12cos/' + 'texture.png', ima)


def makeblack(h, w, value):
    ima = np.full((w,h), value)
    # ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/12cos/' + 'black.png', ima)




def makeimage(h, w, wvcount, phi):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    for i in range(w):
        imaline[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(float(phi)/12.0 + wvcount*float(i)/float(w))))
    print(imaline)
    for j in range(h):
        ima[:, j] = imaline
    # ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/12cos/'+ str(phi + 1) + '_cos.jpg', ima)

# folder = "/home/samir/dblive/cosines/singleshotcosines/"

makeimage(width, height, hfperiods, 0)
makeimage(width, height, hfperiods, 1)
makeimage(width, height, hfperiods, 2)
makeimage(width, height, hfperiods, 3)
makeimage(width, height, hfperiods, 4)
makeimage(width, height, hfperiods, 5)
makeimage(width, height, hfperiods, 6)
makeimage(width, height, hfperiods, 7)
makeimage(width, height, hfperiods, 8)
makeimage(width, height, hfperiods, 9)
makeimage(width, height, hfperiods, 10)
makeimage(width, height, hfperiods, 11)
maketexture(width,height,200)
makeblack(width,height,2)
makeimage(width, height, periods, 20)
makeimage(width, height, periods, 21)
makeimage(width, height, periods, 22)
makeimage(width, height, periods, 23)
makeimage(width, height, periods, 24)
makeimage(width, height, periods, 25)
makeimage(width, height, periods, 26)
makeimage(width, height, periods, 27)
makeimage(width, height, periods, 28)
makeimage(width, height, periods, 29)
makeimage(width, height, periods, 30)
makeimage(width, height, periods, 31)