from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.misc import imread, imresize
import numpy as np
import cv2

COCO_image_path = 'train2014/'
IMG_SIZE = 256
IMG_CHANNEL = 3
#Mean pixel for all images in the set. It is provided by https://github.com/lengstrom/fast-style-transfer
MEAN_PIXEL = np.array([[123.68, 116.779, 103.939]])

#Read image function for style image or test image 
def read_image(filename, BATCH=False, batch_size=64):
    if BATCH:
       img = imread(filename)
       img = imresize(img, (IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
       img_batch = []
       for _ in range(batch_size):
           img_batch.append(img)
       img_batch = np.array(img_batch).astype(np.float32)
       return img_batch
    else:
       img = imread(filename)
       img = imresize(img, (IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
       img_batch = []
       img_batch.append(img)
       img_batch = np.array(img_batch).astype(np.float32)
       return img_batch

#Get batches from the data set. I know there are better methods in Tensorflow to get input data, but I just read them from the prepared list
#for simplicity.
def get_batches(filename, batch_index, batch_size=4):
    lines = open(filename, 'r')
    images = []
    lines = list(lines)
    image_indices = range(len(lines))
    count = 0
    for i in image_indices[batch_index: batch_index + batch_size]:
        if count >= batch_size:
            break
        count += 1
        dirname = lines[i].strip('\n').split()
        img = imread(dirname[0])
        img = imresize(img, (IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
        #The only process for input images is subtracting mean value of each channel.
        if len(img.shape) < 3:
           timg = img
           img = np.zeros((IMG_SIZE, IMG_SIZE, IMG_CHANNEL)).astype(np.float32)
           img[:, :, 0] = timg -  MEAN_PIXEL[0, 0]
           img[:, :, 1] = timg -  MEAN_PIXEL[0, 1]
           img[:, :, 2] = timg -  MEAN_PIXEL[0, 2]
        else:
           img[:, :, 0] = img[:, :, 0] - MEAN_PIXEL[0, 0]
           img[:, :, 1] = img[:, :, 1] - MEAN_PIXEL[0, 1]
           img[:, :, 2] = img[:, :, 2] - MEAN_PIXEL[0, 2]
        '''
        cv2.namedWindow('test win', flags=0)
        cv2.imshow('test win', img)
        cv2.waitKey(0)
        '''
        images.append(img)

    images_np = np.array(images).astype(np.float32)
    batch_index = batch_index + batch_size
    return images_np, batch_index
