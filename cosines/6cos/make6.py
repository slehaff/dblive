import cv2
import numpy as np
from PIL import Image

width = 4096
height = 2732
hfperiods = 120
periods = 12


def maketexture(h, w, value):
    ima = np.full((w,h), value)
    print('texture')
    # ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/6cos/' + 'texture.png', ima)


def makeblack(h, w, value):
    ima = np.full((w,h), value)
    # ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/6cos/' + 'black.png', ima)




def makeimage(h, w, wvcount, phi):
    cima = np.zeros([w,h,3])
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    for i in range(w):
        imaline[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(float(phi)/6.0 + wvcount*float(i)/float(w))))
    print(imaline)
    for j in range(h):
        ima[:, j] = imaline
    # ima = np.transpose(ima)
    cima[:,:,1] = ima

    cv2.imwrite('/home/samir/dblive/cosines/6cos/'+ str(phi + 1) + '_cos.jpg', cima)

# folder = "/home/samir/dblive/cosines/singleshotcosines/"

makeimage(width, height, hfperiods, 0)
makeimage(width, height, hfperiods, 1)
makeimage(width, height, hfperiods, 2)
makeimage(width, height, hfperiods, 3)
makeimage(width, height, hfperiods, 4)
makeimage(width, height, hfperiods, 5)
maketexture(width,height,200)
makeblack(width,height,2)
makeimage(width, height, periods, 20)
makeimage(width, height, periods, 21)
makeimage(width, height, periods, 22)
makeimage(width, height, periods, 23)
makeimage(width, height, periods, 24)
makeimage(width, height, periods, 25)
