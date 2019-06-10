# 输入图片路径， 将图中公式部分加上红框

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import sys
import inference
import image_utils as iu
from PIL import Image, ImageDraw
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

NET = 'os/my_net.ckpt'

def main(img):
    im_init = Image.open(img).convert('RGB')
    im, start = iu.crop_border(img, False)
    im = im.convert('L')
    im_lines, im_lines_words, lines_words, rotated = iu.crop_lines_words(im, False)
    if rotated:
        im_init = im_init.rotate(-90, expand=True)
    lines_words = flat2d(lines_words)
    im_net = np.asarray([np.asarray(x.resize((50, 50)), dtype=np.float32) / 255 for x in lines_words])
    im_net = np.reshape(im_net, [-1, 50, 50, 1])
    result = np.reshape(net_check(im_net), (-1, ))
    # print(result)
    # print(im_lines_words)

# ===========================不合并相邻
    # ind = 0
    # for i in range(len(im_lines_words)):
    #     for j in range(len(im_lines_words[i])):
    #         if result[ind + j] == 1:
    #             x_min = start[0] + im_lines_words[i][j][0]
    #             x_max = start[0] + im_lines_words[i][j][1]
    #             y_min = start[1] + im_lines[i][0]
    #             y_max = start[1] + im_lines[i][1]
    #             draw = ImageDraw.Draw(im_init)
    #             draw.rectangle((x_min, y_min, x_max - 1, y_max - 1), outline=(255, 0, 0), width=2)
    #     ind += len(im_lines_words[i])

# ============================合并相邻
    ind = 0
    for i in range(len(im_lines_words)):
        formu_start = []
        formu_end = []

        leng = len(im_lines_words[i])

        if result[ind] == 1:
            formu_start.append(0)
            if result[ind + 1] == 0:
                formu_end.append(0)

        for j in range(1, leng - 1):
            if result[ind + j] == 1 and result[ind + j - 1] == 0:
                formu_start.append(j)
            if result[ind + j] == 1 and result[ind + j + 1] == 0:
                formu_end.append(j)

        if result[ind + leng - 1] == 1:
            formu_end.append(leng - 1)
            if result[ind + leng - 2] == 0:
                formu_start.append(leng - 1)

        # print(formu_start, formu_end)


        for j in range(len(formu_start)):
            x_min = start[1] + im_lines_words[i][formu_start[j]][0]
            x_max = start[1] + im_lines_words[i][formu_end[j]][1]
            y_min = start[0] + im_lines[i][0]
            y_max = start[0] + im_lines[i][1]
            draw = ImageDraw.Draw(im_init)
            draw.rectangle((x_min-1, y_min-1, x_max, y_max), outline=(255, 0, 0), width=2)

        ind += len(im_lines_words[i])

# ==================================================

    im_init.save('out.png')
    

def net_check(im):
    x = tf.placeholder(tf.float32, [None, 50, 50, 1])
    y = inference.inference(x, None, False, None)
    variable_averages = tf.train.ExponentialMovingAverage(
        0.99)
    saver = tf.train.Saver(tf.global_variables(), variable_averages.variables_to_restore())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, NET)
        pre = sess.run(tf.cast(tf.greater(y, 0), tf.int32), feed_dict={x:im})
        return pre


def flat2d(lis):
    out_lis = lis[0]
    for i in range(1, len(lis)):
        out_lis.extend(lis[i])
    return out_lis


main(sys.argv[1])


