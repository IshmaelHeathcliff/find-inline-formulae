import tensorflow as tf
import inference
import os
import numpy as np
import random
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_PATH = 'test/train.tfrecords'
INPUT_SIZE = 50
BATCH_SIZE = 5
DATA_NUM = 50
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99



def train(filename):

    # 定义输出为4维矩阵的placeholder
    x_train, y_train = get_data(filename)
    x = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization = tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                            global_step,
                                            DATA_NUM,
                                            LEARNING_RATE_DECAY,
                                            staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(TRAINING_STEPS):
            xi, yi = sess.run([x_train, y_train])
            xs = np.reshape(xi, (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1))
            ys = np.reshape(yi, (BATCH_SIZE, 1))
            _, loss_value, step, yo = sess.run([train_op, loss, global_step, y], feed_dict={x: xs, y_: ys})

            if i % 100 == 0:
                print(
                    "After %d training step(s), loss on training batch is %g."
                    % (step, loss_value))
                print('y, y_:', yo, yi)

        coord.request_stop()
        coord.join(threads)


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

    img = tf.reshape(img, (INPUT_SIZE, INPUT_SIZE))
    img = tf.cast(img, tf.float32) * (1. / 255)

    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=BATCH_SIZE, capacity= 100, num_threads= 2, min_after_dequeue= 10)

    
    return img_batch, label_batch

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def main(argv=None):
    train(INPUT_PATH)


if __name__ == '__main__':
    main()
