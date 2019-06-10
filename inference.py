# 神经网络结构， conv3 --> pool --> conv3 --> pool --conv3 --> spp --> fc1 --> fc2

import tensorflow as tf
import math

OUTPUT_NODE = 2

NUM_CHANNELS = 1
NUM_LABELS = 1

CONV1_DEEP = 32
CONV1_SIZE = 3

CONV2_DEEP = 32
CONV2_SIZE = 3

CONV3_DEEP = 64
CONV3_SIZE = 3

CONV4_DEEP = 128
CONV4_SIZE = 3

BINS = [3, 2, 1]
FC1_SIZE = 14*128
FC2_SIZE = 128


def inference(input_tensor, avg, train, regularizer):
    with tf.variable_scope('layer1-conv1', reuse=tf.AUTO_REUSE):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.contrib.layers.variance_scaling_initializer())
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        if avg is not None:
            conv1_weights = avg.average(conv1_weights)
            conv1_biases = avg.average(conv1_biases)

        conv1 = tf.nn.conv2d(input_tensor,
                             conv1_weights,
                             strides=[1, 1, 1, 1],
                             padding='VALID')
        bn1 = tf.layers.batch_normalization(conv1, training=train)
        prelu1 = prelu(tf.nn.bias_add(bn1, conv1_biases), avg)

    with tf.variable_scope('layer2-conv2', reuse=tf.AUTO_REUSE):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.contrib.layers.variance_scaling_initializer())
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        if avg is not None:
            conv2_weights = avg.average(conv2_weights)
            conv2_biases = avg.average(conv2_biases)

        conv2 = tf.nn.conv2d(prelu1,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='VALID')
        bn2 = tf.layers.batch_normalization(conv2, training=train)
        prelu2 = prelu(tf.nn.bias_add(bn2, conv2_biases), avg)

    with tf.variable_scope('layer3-pool1', reuse=tf.AUTO_REUSE):
        pool1 = tf.nn.max_pool(prelu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

    with tf.variable_scope("layer4-conv3", reuse=tf.AUTO_REUSE):
        conv3_weights = tf.get_variable(
            "weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
            initializer=tf.contrib.layers.variance_scaling_initializer())
        conv3_biases = tf.get_variable(
            "bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))

        if avg is not None:
            conv3_weights = avg.average(conv3_weights)
            conv3_biases = avg.average(conv3_biases)

        conv3 = tf.nn.conv2d(pool1,
                             conv3_weights,
                             strides=[1, 1, 1, 1],
                             padding='VALID')
        bn3 = tf.layers.batch_normalization(conv3, training=train)
        prelu3 = prelu(tf.nn.bias_add(bn3, conv3_biases), avg)

    with tf.variable_scope('layer5-pool2', reuse=tf.AUTO_REUSE):
        pool2 = tf.nn.max_pool(prelu3, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

    with tf.variable_scope("layer6-conv4", reuse=tf.AUTO_REUSE):
        conv4_weights = tf.get_variable(
            "weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
            initializer=tf.contrib.layers.variance_scaling_initializer())
        conv4_biases = tf.get_variable(
            "bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))

        if avg is not None:
            conv4_weights = avg.average(conv4_weights)
            conv4_biases = avg.average(conv4_biases)

        conv4 = tf.nn.conv2d(pool2,
                             conv4_weights,
                             strides=[1, 1, 1, 1],
                             padding='VALID')
        bn4 = tf.layers.batch_normalization(conv4, training=train)
        prelu4 = prelu(tf.nn.bias_add(bn4, conv4_biases), avg)

    with tf.variable_scope('layer7-spp', reuse=tf.AUTO_REUSE):
        spp = Spp_layer(prelu4, BINS)

    with tf.variable_scope('layer8-fc1', reuse=tf.AUTO_REUSE):
        fc1_weights = tf.get_variable(
            "weight", [FC1_SIZE, FC2_SIZE],
            initializer=tf.contrib.layers.variance_scaling_initializer())
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC2_SIZE],
                                    initializer=tf.constant_initializer(0.1))

        if avg is not None:
            fc1_weights = avg.average(fc1_weights)
            fc1_biases = avg.average(fc1_biases)

        fc1 = prelu(tf.matmul(spp, fc1_weights) + fc1_biases, avg)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer9-fc2', reuse=tf.AUTO_REUSE):
        fc2_weights = tf.get_variable(
            "weight", [FC2_SIZE, NUM_LABELS],
            initializer=tf.contrib.layers.variance_scaling_initializer())
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS],
                                    initializer=tf.constant_initializer(0.1))

        if avg is not None:
            fc2_weights = avg.average(fc2_weights)
            fc2_biases = avg.average(fc2_biases)

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


def Spp_layer(feature_map, bins):
    bs, x, y, _ = feature_map.get_shape().as_list()
    batch_size = tf.shape(feature_map)[0]
    pooling_out_all = []

    for layer in range(len(bins)):
        k_size = math.ceil(x / bins[layer])
        stride = math.floor(x / bins[layer])
        pooling_out = tf.nn.max_pool(feature_map, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='VALID')
        pooling_out_resized = tf.reshape(pooling_out, [batch_size, -1])
        pooling_out_all.append(pooling_out_resized)

    feature_map_out = tf.concat(axis=1, values=pooling_out_all)
    return feature_map_out


def prelu(_x, avg):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                        initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
    if avg is not None:
        alphas = avg.average(alphas)  

    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg