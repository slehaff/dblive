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
for i in range(0,10):
    folder1 = '/home/samir/Desktop/blender/pycode/inputscans/render'+ str(i)
    folder2 = '/home/samir/Desktop/blender/pycode/inputscans/render'+str(i)

    monohigh = np.zeros((H, W), dtype=np.float64)

    high = folder1 + '/unwrap1.npy'
    
    print(high)
    # colorhigh = cv2.imread(high, 1)
    # monohigh1 = make_grayscale(colorhigh)
    # monohigh1 = 255*normalize_image(monohigh1)
    monohigh1 = np.load(high)
    monohigh1 = monohigh1
    print('unwrange=', np.ptp(monohigh1), np.max(monohigh1), np.min(monohigh1) )
    
    # high = folder1 + '/depth.png'
    high1 = folder1 + '/nnkdata.npy'
    print(high1)
    # colorhigh = cv2.imread(high1, 1)
    # monohigh3 = make_grayscale(colorhigh)
    # monohigh3 = 255*normalize_image(monohigh3)
    colorhigh = 4*(np.load(high1))
    print('unwrange=', np.ptp(colorhigh), np.max(colorhigh), np.min(colorhigh) )

    monohigh3 = colorhigh

    high = folder1 + '/nnkdata.png'
    print(high)
    colorhigh = cv2.imread(high, 1)
    monohigh2 = make_grayscale(colorhigh)
    monohigh2 = 255*normalize_image(monohigh2)
    # monohigh1 = np.load(high)
    
    
    # high = folder1 + '/depth.png'
    high1 = folder1 + '/nnkunwrap.png'
    print(high1)
    colorhigh = cv2.imread(high1, 1)
    monohigh4 = make_grayscale(colorhigh)
    monohigh4 = 255*normalize_image(monohigh4)


    # colorhigh = np.load(high1)
    # monohigh3 = colorhigh

    # high = folder2 + '/image0.png'
    # colorhigh = cv2.imread(high, 1)
    # colorhigh = resize(colorhigh, W, H)
    # monohigh2 = make_grayscale(colorhigh)
    # monohigh2 = 255*normalize_image(monohigh2)
    # high = folder2 + '/im_wrap1.png'
    # colorhigh = cv2.imread(high, 1)
    # colorhigh = resize(colorhigh, W, H)
    # monohigh4 = make_grayscale(colorhigh)
    # monohigh4 = 255*normalize_image(monohigh4)


    for a in range(0,160,20):
        plt.plot(
        x, monohigh1[:,a],'r',
        x, monohigh3[:,a],'b')
        plt.ylabel(str(i)) 
        plt.show()
        plt.plot(
        x, monohigh2[:,a],'r',
        x, monohigh4[:,a],'b')
        plt.ylabel(str(i)) 
        plt.show()