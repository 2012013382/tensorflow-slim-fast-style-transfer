#coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
import numpy as np
from os.path import join
from scipy.misc import imread, imresize, imsave
TEST_IMAGE_PATH = 'test/test.jpg'
MODEL_PATH = 'model/trained_model/cubist/model.ckpt-41391'
IMAGE_SAVE_PATH = './'
#Read test image
test_img = imread(TEST_IMAGE_PATH)
t_shape = test_img.shape
test_img = imresize(test_img, (t_shape[0], t_shape[1], t_shape[2]))
test_imgg = []
test_imgg.append(test_img)
test_imgg = np.array(test_imgg).astype(np.float32)
with tf.Graph().as_default():
    test_image = tf.placeholder(tf.float32, [None, t_shape[0], t_shape[1], t_shape[2]])
    generated_image = model.generator(test_image, training=False)
    squeezed_generated_image = tf.squeeze(generated_image, [0])

    restorer = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restorer.restore(sess, MODEL_PATH)
        styled_image = sess.run(squeezed_generated_image, feed_dict={test_image: test_imgg})
        styled_imgae = (styled_image + 1.0) * 127.5
        imsave(join(IMAGE_SAVE_PATH, 'test.jpg'), np.squeeze(styled_image))
