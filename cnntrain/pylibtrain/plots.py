# This module plots a row in 2 different files for comparing fringe and wral patterns between different files


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
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 28)
    draw.text((200, 200), text, (255), font=font)
    img.save(filename)
    folder = "/home/samir/db3/prototype/pylib/oralcosines/"



def resize(img, w,h):
    print(img.shape)
    img = img[0:160,0:160]
    print( img.shape )
    return(img)


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img



def makemonohigh(folder):
    high = folder + '/blenderimage0.png'
    colorhigh = cv2.imread(high, 1)
    monohigh = make_grayscale(colorhigh)
    cv2.imwrite(folder+'monohigh.png', monohigh)
    return
folder = '/home/samir/Desktop/blender/pycode/stitch2files/'
bfolder = '/home/samir/Desktop/blender/pycode/stitch2/'
folder1 = '/home/samir/Desktop/blender/pycode/stitch2/render'
folder2 = '/home/samir/Desktop/blender/pycode/stitch2/render'

monohigh = np.zeros((H, W), dtype=np.float64)

for i in range(len(os.listdir(bfolder))):
    my_folder = folder+'neural/'  +'n'+ str(i)+'/'
    print(my_folder)
    if not os.path.exists(my_folder):
        print('absent!')
        os.makedirs(my_folder)

    high = folder2 + str(i)+ '/unwrap1.png'
    colorhigh = cv2.imread(high, 1)
    colorhigh = resize(colorhigh, W, H)
    monohigh1 = make_grayscale(colorhigh)

    high = folder1 +str(i)+  '/nnkdata.png'
    colorhigh = cv2.imread(high, 1)
    colorhigh = resize(colorhigh, W, H)
    monohigh2 = make_grayscale(colorhigh)


    x = range(160)

    for j in range(0,160,10):
        fig = plt.figure()
        plt.plot(x, monohigh1[:,j],
        x, monohigh2[:,j])
        plt.ylabel(str(j))
        fig.savefig(my_folder+ 'plot'+str(j)+'.png')
        plt.close(fig) 
        # plt.show()




def repairK(wrapfile, kfile, krepfile):
    high = folder2 + str(i)+ wrapfile
    wdata = cv2.imread(high, 1)

    k = folder2 + str(i)+ kfile
    kdata = cv2.imread(k, 1)

    for j in range(160):
        for i in range(160):
            if wdata[i,j] < lowthresh :
                lowthresh == True
