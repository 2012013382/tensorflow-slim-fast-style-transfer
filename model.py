import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
'''
This file is only include the generator net and loss net(vgg_16)
We apply instance normalization and resize conv2d.
We utilize slim to build network for its simplicity
Function instance_norm, resize_conv2d and gram are provided by https://github.com/hzy46/fast-neural-style-tensorflow
'''
def conv2d_slim(x, filter_num, kernel_size, strides, name):
    return slim.conv2d(x, filter_num, [kernel_size, kernel_size], stride=strides, weights_regularizer=slim.l2_regularizer(1e-6), biases_regularizer=slim.l2_regularizer(1e-6), padding='SAME', activation_fn=None, scope=name)

def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def residual(x, filter_num, kernel_size, strides, name):
    with tf.variable_scope(name):
        conv1 = conv2d_slim(x, filter_num, kernel_size, strides, 'conv1')
        conv2 = conv2d_slim(tf.nn.relu(conv1), filter_num, kernel_size, strides, 'conv2')
        residual = x + conv2
        return residual

def resize_conv2d(x, filters_num, kernel_size, strides, training, name):
    with tf.variable_scope(name):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return conv2d_slim(x_resized, filters_num, kernel_size, strides, 'conv1')


def generator(image, training):
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    with tf.variable_scope('generator'):
        conv1 = tf.nn.relu(instance_norm(conv2d_slim(image, 32, 9, 1, 'conv1')))
        conv2 = tf.nn.relu(instance_norm(conv2d_slim(conv1, 64, 3, 2, 'conv2')))
        conv3 = tf.nn.relu(instance_norm(conv2d_slim(conv2, 128, 3, 2, 'conv3')))
        res1 = residual(conv3, 128, 3, 1, 'res1')
        res2 = residual(res1, 128, 3, 1, 'res2')
        res3 = residual(res2, 128, 3, 1, 'res3')
        res4 = residual(res3, 128, 3, 1, 'res4')
        res5 = residual(res4, 128, 3, 1, 'res5')
        deconv1 = tf.nn.relu(instance_norm(resize_conv2d(res5, 64, 3, 1, training, 'deconv1')))
        deconv2 = tf.nn.relu(instance_norm(resize_conv2d(deconv1, 32, 3, 1, training, 'deconv2')))
        deconv3 = tf.nn.tanh(instance_norm(conv2d_slim(deconv2, 3, 9, 1, 'deconv3')))
        #re-vlaue to [0, 255]
        y = (deconv3 + 1.0) * 127.5
        height = tf.shape(y)[1]
        width = tf.shape(y)[2]
	y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))
        return y

#Loss model Vgg 16 provided by slim.
def loss_model(x):
    #x = x / 127.5 - 1
    logits, endpoints_dict = nets.vgg.vgg_16(x, spatial_squeeze=False)
    return logits, endpoints_dict

#content loss
def content_loss(endpoints_mixed, content_layers):
    loss = 0
    for layer in content_layers:
        A, B, _ = tf.split(endpoints_mixed[layer], 3, 0)
        size = tf.size(A)
        loss += tf.nn.l2_loss(A - B) * 2 / tf.to_float(size)
    return loss

#style loss
def style_loss(endpoints_mixed, style_layers):
    loss = 0
    for layer in style_layers:
        _, B, C = tf.split(endpoints_mixed[layer], 3, 0)
        size = tf.size(B) 
        loss += tf.nn.l2_loss(gram(B) - gram(C)) * 2 / tf.to_float(size)
    return loss

#Gram
def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    features = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(features, features, transpose_a=True) / tf.to_float(width * height * num_filters)
    return grams
