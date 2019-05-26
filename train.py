import tensorflow as tf
import inference
import os
import numpy as np

INPUT_PATH = ''
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99


def train(data):
    data_num = len(data)
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [None, None, None, inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE],
                        name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               len(data),
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
            xs = data[i % data_num][0]
            ys = data[i % data_num][1]

            for i in range(len(xs)):
                xs_i = np.asarray(xs[i])
                xs[i] = np.reshape(
                    xs_i,
                    (1, xs_i.shape[0], xs_i.shape[1], inference.NUM_CHANNELS))

                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={
                                                   x: xs_i,
                                                   y_: [ys[i]]
                                               })

            if i % 10 == 0:
                print(
                    "After %d training step(s), loss on training batch is %g."
                    % (step, loss_value))


def data_process(datapath):
    database = np.load(datapath)
    data = database[:, 2:4].tolist()
    for i in range(len(data)):
        data[i][0] = np.asarray(data[i][0]).flatten().tolist()
        data[i][1] = np.asarray(data[i][1]).flatten().tolist()
    return data


def main(argv=None):
    data = data_process(INPUT_PATH)
    train(data)


if __name__ == '__main__':
    main()
