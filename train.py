# 神经网络训练主体程序

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import inference
import numpy as np
from math import exp

tf.logging.set_verbosity(tf.logging.ERROR)

TRAIN_DATA = 'dataset/train/sq/train.tfrecords-*'
TEST_DATA = 'dataset/test/sqtest.tfrecords'
INPUT_SIZE = 50
BATCH_SIZE = 100
TEST_BATCH_SIZE = 1000
DATA_NUM =  100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.9
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99


def train():

    # 训练数据输入
    x_train_batch, y_train_batch = get_data(TRAIN_DATA, BATCH_SIZE)
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')
    global_step = tf.Variable(0, trainable=False)

    # 正则项与网络输出
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, None, True, regularizer)

    # 滑动平均操作
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step / 10)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 测试数据输入
    x_test_batch, y_test_batch = get_data(TEST_DATA, TEST_BATCH_SIZE)
    y_t = inference.inference(x, variable_averages, False, None)
    preds = tf.reshape(tf.cast(tf.greater(y_t, 0.), tf.float32), [TEST_BATCH_SIZE, ])
    labels = tf.reshape(y_, [TEST_BATCH_SIZE, ])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    # 损失函数
    cross_entropy_mean = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=y, targets=y_, pos_weight=0.5))
    loss = cross_entropy_mean  + tf.add_n(tf.get_collection('losses')) # 损失加上正则化

    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                            global_step,
                                            DATA_NUM,
                                            LEARNING_RATE_DECAY,
                                            staircase=True)

    # 训练过程定义
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

        acs = []
        pres = []
        recs = []
        f1s = []
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
                tp = tn = fp = fn = ac = 0
                x_test, y_test = sess.run([x_test_batch, y_test_batch])
                lab = y_test
                x_test = np.reshape(x_test, [TEST_BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1])
                y_test = np.reshape(y_test, [TEST_BATCH_SIZE, 1])
                accu, pred= sess.run([accuracy, preds], feed_dict={x: x_test, y_:y_test})
                pred_t = np.where(pred == 1)[0]
                pred_f = np.where(pred == 0)[0]
                lab_t = np.where(lab == 1)[0]
                lab_f = np.where(lab == 0)[0]
                tp += same_num(pred_t, lab_t)
                fp += same_num(pred_t, lab_f)
                tn += same_num(pred_f, lab_f)
                fn += same_num(pred_f, lab_t)
                ac += accu

                acs.append(ac)
                print('accuracy:', ac)
                pres.append(tp / (tp + fp + exp(-10)))
                print('precision:', tp / (tp + fp + exp(-10)))
                recs.append(tp / (tp + fn + exp(-10)))
                print('recall:', tp / (tp + fn + exp(-10)))
                f1s.append(2*tp / (2*tp + fp + fn + exp(-10)))
                print('f1:', 2*tp / (2*tp + fp + fn + exp(-10)))
        
        print('accuracys:', acs)
        print('precisions:', pres)
        print('recalls:', recs)
        print('f1s:', f1s)

        tp = tn = fp = fn = ac = 0
        for i in range(24):
            x_test, y_test = sess.run([x_test_batch, y_test_batch])
            x_test = np.reshape(x_test, [TEST_BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1])
            y_test = np.reshape(y_test, [TEST_BATCH_SIZE, 1])
            accu, pred, lab = sess.run([accuracy, preds, labels], feed_dict={x: x_test, y_:y_test})
            pred_t = np.where(pred == 1)[0]
            pred_f = np.where(pred == 0)[0]
            lab_t = np.where(lab == 1)[0]
            lab_f = np.where(lab == 0)[0]
            tp += same_num(pred_t, lab_t)
            fp += same_num(pred_t, lab_f)
            tn += same_num(pred_f, lab_f)
            fn += same_num(pred_f, lab_t)
            ac += accu
        print('accuracy:', ac / 24)
        print('precision:', tp / (tp + fp + exp(-10)))
        print('recall:', tp / (tp + fn + exp(-10)))
        print('f1:', 2*tp / (2*tp + fp + fn + exp(-10)))
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

    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=bs, capacity=1000, num_threads=2, min_after_dequeue= 10)

    
    return img_batch, label_batch

def same_num(a, b):
    result = 0
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                result += 1
                break
    return result

def main(argv=None):
    train()


if __name__ == '__main__':
    main()
