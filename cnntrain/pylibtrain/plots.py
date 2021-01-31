# This module plots a row in 2 different files for comparing fringe and wral patterns between different files
# The module is set manually to produce neural plots in "neural" folder or analytic plots in "analytic" folder

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


H=160
W= 160

def addtext(filename, text):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 5)
    draw.text((200, 200), text, (255), font=font)
    img.save(filename)
    folder = "/home/samir/db3/prototype/pylib/oralcosines/"

def denstext(filename):
    print(filename)
    img = cv2.imread(filename)
    img = make_grayscale(img)
    print(img.size)
    for u in range(0,W,5):
        for v in range(0,H,5):
            sum=0
            for i in range(5):
                for j in range(5):
                    sum= sum+img[u+i,v+j]
            sum= np.round(sum/(25*9))
            print(u,v,sum)

def resize(img, w,h):
    print(img.shape)
    img = img[0:160,0:160]
    print( img.shape )
    return(img)


def make_grayscale(img):
    # Transform color image to 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img



def makemonohigh(folder):
    high = folder + '/blenderimage0.png'
    colorhigh = cv2.imread(high, 1)
    monohigh = make_grayscale(colorhigh)
    cv2.imwrite(folder+'monohigh.png', monohigh)
    return



def myplot():
    # folder = '/home/samir/Desktop/blender/pycode/stitchfiles/'
    # bfolder = '/home/samir/Desktop/blender/pycode/stitch/'
    # folder1 = '/home/samir/Desktop/blender/pycode/stitch/render'
    # folder2 = '/home/samir/Desktop/blender/pycode/stitch/render'

    # monohigh = np.zeros((H, W), dtype=np.float64)

    for i in range(len(os.listdir(bfolder))):
        my_folder = folder+'analytic/'  +'a'+ str(i)+'/'
        print(my_folder)
        if not os.path.exists(my_folder):
            print('absent!')
            os.makedirs(my_folder)

        high = folder2 + str(i)+ '/im_wrap1.png'
        colorhigh = cv2.imread(high, 1)
        colorhigh = resize(colorhigh, W, H)
        monohigh1 = make_grayscale(colorhigh)

        high = folder1 +str(i)+  '/kdata.png'
        colorhigh = cv2.imread(high, 1)
        colorhigh = resize(colorhigh, W, H)
        monohigh2 = make_grayscale(colorhigh)


        x = range(160)

        for j in range(0,160,20):
            fig = plt.figure()
            plt.plot(x, monohigh1[:,j],
            x, monohigh2[:,j])
            plt.ylabel(str(j))
            fig.savefig(my_folder+ 'plot'+str(j)+'.png')
            plt.close(fig) 
            plt.show()


folder1 = '/home/samir/Desktop/blender/pycode/stitch/render0/kdata.png'

denstext(folder1)

def repairK(wrapfile, kfile, krepfile):
    high = folder2 + str(i)+ wrapfile
    wdata = cv2.imread(high, 1)

    k = folder2 + str(i)+ kfile
    kdata = cv2.imread(k, 1)

    for j in range(160):
        for i in range(160):
            if ((wdata[i,j] - wdata[i-1,j]<0) and (wdata[i+1,j] - wdata[i,j]>0)):  #bottom spike
                kdata[i,j] = kdata[i-1,j]+ 7


