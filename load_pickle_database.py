import pickle
import numpy as np
from skimage import color

def load_images_labels(filepath):
    data = pickle.load(open(filepath, 'rb'));

    images_read = data[0];
    images = preprocess(images_read);
#     print(images.shape)
    labels = data[1];
#     print(np.nonzero(labels))
#     labels = np.asarray(labels)
#     print(images[0].shape)
    return [images, labels] #images is a 2d array (nSamples,60*40)

def preprocess(images_read):
    #images is an array (nSamples,60,40,3)
    [nSamples,height,width,RGB] = images_read.shape
    grayImages = color.rgb2gray(images_read)
    #grayImages is an array (nSamples,60,40)
    flattenedImages = grayImages.reshape(nSamples,height*width)
    #flattenedImages is a 2d array (nSamples,60*40)
    return flattenedImages
    