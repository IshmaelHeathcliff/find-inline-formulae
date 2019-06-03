# 神经网络训练主体程序

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import inference
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

TRAIN_DATA = 'train.tfrecords-1'
TEST_DATA = 'test.tfrecords'
INPUT_SIZE = 50
BATCH_SIZE = 100
TEST_BATCH_SIZE = 500
DATA_NUM =  1000
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.9
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99


def train():

    # 训练数据输入
    x_train_batch, y_train_batch = get_data(TRAIN_DATA, BATCH_SIZE)
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 测试数据输入
    x_test_batch, y_test_batch = get_data(TEST_DATA, TEST_BATCH_SIZE)
    y_t = inference.inference(x, False, None)
    preds = tf.reshape(tf.cast(tf.greater(y_t, 0.), tf.float32), [TEST_BATCH_SIZE, ])
    labels = tf.reshape(y_, [TEST_BATCH_SIZE, ])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
    test_cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_t, labels=y_))
    loss = cross_entropy_mean  + tf.add_n(tf.get_collection('losses')) # 损失加上正则化

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                            global_step,
                                            DATA_NUM,
                                            LEARNING_RATE_DECAY,
                                            staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    with tf.control_dependencies([variables_averages_op]): # 添加update_ops
        train_op = tf.group([train_steps, update_ops])

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.local_variables_initializer().run() # match_filenames_once 初始化
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(TRAINING_STEPS):
            x_train, y_train = sess.run([x_train_batch, y_train_batch])
            x_train = np.reshape(x_train, [BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1])
            y_train = np.reshape(y_train, [BATCH_SIZE, 1])
            _, loss_value, step, yo = sess.run([train_op, loss, global_step, y], feed_dict={x: x_train, y_: y_train})

            if i % 1000 == 0:
                print(
                    "After %d training step(s), loss on training batch is %g."
                    % (step, loss_value))
                # print('y, y_:', yo, y_train)

                x_test, y_test = sess.run([x_test_batch, y_test_batch])
                x_test = np.reshape(x_test, [TEST_BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1])
                y_test = np.reshape(y_test, [TEST_BATCH_SIZE, 1])
                test_loss, pre, lab, accu = sess.run([test_cross_entropy_mean, preds, labels, accuracy], feed_dict={x: x_test, y_:y_test})
                print("Test accuracy:", accu)
                print("Test loss:", test_loss)
                # print("preditions:", pre)
                # print("labels:", lab)

        saver.save(sess, './my_net.ckpt')

        coord.request_stop()
        coord.join(threads)

def get_data(filename, bs):
    files = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(files)
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

    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=bs, capacity= 1000, num_threads= 2, min_after_dequeue= 10)

    
    return img_batch, label_batch


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
