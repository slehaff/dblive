
# example of zoom image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img1 = load_img('/home/samir/dblive/cnntrain/pylibtools/images/image0.png')
img2 = load_img('/home/samir/dblive/cnntrain/pylibtools/images/im_wrap1.png')
# convert to numpy array
data1 = img_to_array(img1)
# expand dimension to one sample
samples1 = expand_dims(data1, 0)
print(samples1.shape)

# convert to numpy array
data2 = img_to_array(img2)
# expand dimension to one sample
samples2 = expand_dims(data2, 0)
print(samples2.shape)
# create image data augmentation generator
datagen = ImageDataGenerator(zoom_range=[0.5,1])
# prepare iterator
it = datagen.flow(samples2,samples1, batch_size=1)
# generate samples and plot
print(len(it))
for i in range(24):
	# define subplot
	pyplot.subplot(3,8,i+1)
	# generate batch of images
	batch = it.next()[0]
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
