import tensorflow as tf
import math

OUTPUT_NODE = 2

NUM_CHANNELS = 1
NUM_LABELS = 2

CONV1_DEEP = 32
CONV1_SIZE = 3

CONV2_DEEP = 64
CONV2_SIZE = 3

BINS = [3, 2, 1]
FC_SIZE = 14 * 64


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,
                             conv1_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope("layer2-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(conv1,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layer3-spp'):
        spp = Spp_layer(relu2, BINS)

    with tf.variable_scope('layer4-fc'):
        fc_weights = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc_weights))
        fc_biases = tf.get_variable("bias", [NUM_LABELS],
                                    initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(spp, fc_weights) + fc_biases

    return logit


def Spp_layer(feature_map, bins):
    batch_size, x, y, _ = feature_map.get_shape().as_list()
    pooling_out_all = []
    print("shape of pool2:", x, y)

    for layer in range(len(bins)):
        k_size_x = math.ceil(x / bins[layer])
        k_size_y = math.ceil(y / bins[layer])
        stride_x = math.floor(x / bins[layer])
        stride_y = math.floor(y / bins[layer])
        print("kernel and stride size:", k_size_x, k_size_y, stride_x, stride_y)
        pooling_out = tf.nn.max_pool(feature_map,
                                     ksize=[1, k_size_x, k_size_y, 1],
                                     strides=[1, stride_x, stride_y, 1],
                                     padding='VALID')
        pooling_out_resized = tf.reshape(pooling_out, [batch_size, -1])
        print("spp size:", pooling_out_resized.get_shape().as_list())
        pooling_out_all.append(pooling_out_resized)

    feature_map_out = tf.concat(axis=1, values=pooling_out_all)
    return feature_map_out
