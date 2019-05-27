import tensorflow as tf
import inference
import os
import numpy as np
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

INPUT_PATH = 'train.tfrecords'
DATA_NUM = 50
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 500
MOVING_AVERAGE_DECAY = 0.99



def train(filename):

    # 定义输出为4维矩阵的placeholder
    x_train, y_train = get_data(filename)
    print(x_train.get_shape())
    x = tf.placeholder(tf.float32, (1, x_train.get_shape()[0], x_train.get_shape()[1], 1), name='x-input')
    y_ = tf.placeholder(tf.float32, (1, 1), name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.cast(tf.reshape(y_, (1,)), tf.int32))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                            global_step,
                                            DATA_NUM,
                                            LEARNING_RATE_DECAY,
                                            staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = sess.run([x_train, y_train])
            xs = np.reshape(xs, (1, xs.shape[0], xs.shape[1], 1))
            ys = np.reshape(ys, (1, 1))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 10 == 0:
                print(
                    "After %d training step(s), loss on training batch is %g."
                    % (step, loss_value))


def get_data(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
      features={
      'label': tf.FixedLenFeature([], tf.int64),
      'img' : tf.FixedLenFeature([], tf.string)})
    
    img = tf.decode_raw(features['img'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    
    return img, label


def main(argv=None):
    train(INPUT_PATH)


if __name__ == '__main__':
    main()
