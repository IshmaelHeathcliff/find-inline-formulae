# not working

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import sys
import inference
import tensorflow as tf
from math import exp

tf.logging.set_verbosity(tf.logging.ERROR)

INPUT_SIZE = 50
TEST_DATA = 'dataset/train.tfrecords-4'
TEST_BATCH_SIZE = 1000
MOVING_AVERAGE_DECAY = 0.99


def evaluate():
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

    # 滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY)

    # 测试数据输入
    x_test_batch, y_test_batch = get_data(TEST_DATA, TEST_BATCH_SIZE)
    y_t = inference.inference(x, None, False, None)
    preds = tf.reshape(tf.cast(tf.greater(y_t, 0.), tf.float32), [TEST_BATCH_SIZE, ])
    labels = tf.reshape(y_, [TEST_BATCH_SIZE, ])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    saver = tf.train.Saver(tf.global_variables(), variable_averages.variables_to_restore())
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, 'my_net.ckpt')
        
        tp = tn = fp = fn = ac = 0
        for i in range(10):
            x_test, y_test = sess.run([x_test_batch, y_test_batch])
            x_test = np.reshape(x_test, [TEST_BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1])
            y_test = np.reshape(y_test, [TEST_BATCH_SIZE, 1])
            accu, pred, lab= sess.run([accuracy, preds, labels], feed_dict={x: x_test, y_:y_test})
            pred_t = np.where(pred == 1)[0]
            pred_f = np.where(pred == 0)[0]
            lab_t = np.where(lab == 1)[0]
            lab_f = np.where(lab == 0)[0]
            tp += same_num(pred_t, lab_t)
            fp += same_num(pred_t, lab_f)
            tn += same_num(pred_f, lab_f)
            fn += same_num(pred_f, lab_t)
            ac += accu
        print('accuracy:', ac / 10)
        print('precision:', tp / (tp + fp + exp(-10)))
        print('recall:', tp / (tp + fn + exp(-10)))
        print('f1:', 2*tp / (2*tp + fp + fn + exp(-10)))

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

    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=bs, capacity=1000, num_threads=2, min_after_dequeue= 10)
    
    return img_batch, label_batch

def same_num(a, b):
    result = 0
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                result += 1
                continue
    return result

evaluate()