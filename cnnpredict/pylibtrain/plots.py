# This module plots a row in 2 different files for comparing fringe and wral patterns between different files


import matplotlib.pyplot as plt
import cv2
import numpy as np

H=160
W= 160


def resize(img, w,h):
    print(img.shape)
    img = img[0:160,0:160]
    print( img.shape )
    return(img)



def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img



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

# folder1 = '/home/samir/Desktop/blender/pycode/inputscans/render1'
# folder2 = '/home/samir/Desktop/blender/pycode/inputscans/render1'

# monohigh = np.zeros((H, W), dtype=np.float64)

# high = folder1 + '/cnnwrap1.png'
# colorhigh = cv2.imread(high, 1)
# colorhigh = resize(colorhigh, W, H)
# monohigh1 = make_grayscale(colorhigh)
# monohigh1 = 255*normalize_image(monohigh1)

# high = folder1 + '/blenderimage0.png'
# colorhigh = cv2.imread(high, 1)
# colorhigh = resize(colorhigh, W, H)
# monohigh2 = make_grayscale(colorhigh)
# monohigh2 = 255*normalize_image(monohigh2)


# x = range(160)

# for i in range(0,160,20):
#     plt.plot(x, monohigh1[:,i],
#     x, monohigh2[:,i])
#     plt.ylabel(str(i)) 
#     plt.show()


x = range(160)
for i in range(0,250,20):
    folder1 = '/home/samir/Desktop/blender/pycode/coldepthplanes/render'+ str(i)
    folder2 = '/home/samir/Desktop/blender/pycode/coldepthplanes/render'+str(i)

    monohigh = np.zeros((H, W), dtype=np.float64)

    high = folder1 + '/image0.png'
    colorhigh = cv2.imread(high, 1)
    colorhigh = resize(colorhigh, W, H)
    monohigh1 = make_grayscale(colorhigh)
    monohigh1 = 255*normalize_image(monohigh1)
    high1 = folder1 +'/unwrap.npy'
    monohigh1 = np.load(high1)
    
    
    high = folder1 + '/kdata.npy'
    # colorhigh = cv2.imread(high, 1)
    # colorhigh = resize(colorhigh, W, H)
    # monohigh3 = make_grayscale(colorhigh)
    # monohigh3 = 255*normalize_image(monohigh3)
    colorhigh = np.load(high)
    monohigh3 = colorhigh

    high = folder2 + '/image0.png'
    colorhigh = cv2.imread(high, 1)
    colorhigh = resize(colorhigh, W, H)
    monohigh2 = make_grayscale(colorhigh)
    monohigh2 = 255*normalize_image(monohigh2)
    high = folder2 + '/im_wrap1.png'
    colorhigh = cv2.imread(high, 1)
    colorhigh = resize(colorhigh, W, H)
    monohigh4 = make_grayscale(colorhigh)
    monohigh4 = 255*normalize_image(monohigh4)

    plt.plot(
    x, monohigh1[:,80],'g',
    x, monohigh3[:,80],'m')
    plt.ylabel(str(i)) 
    plt.show()
    plt.plot(
    x, monohigh2[:,80],'r',
    x, monohigh4[:,80],'b')
    plt.ylabel(str(i)) 
    plt.show()