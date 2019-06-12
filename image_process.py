# 批量处理图片获得数据tfrecords

import tensorflow as tf
import numpy as np
import sys
import os
import image_utils as iu
from PIL import Image

IMG_DIR = 'dataset/2003/Train'
TEST_IMG_DIR = 'dataset/test'
OUT_PATH = 'train.tfrecords'


def _init64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def flat2d(lis):
    out_lis = lis[0]
    for i in range(1, len(lis)):
        out_lis.extend(lis[i])
    return out_lis


def image_words_prep(IMG_DIR, train=True):
    os.chdir(IMG_DIR)

    dirs = [x[1] for x in os.walk('./')][0]
    images = []
    labels = []
    count = 0
    num = 1
    for di in dirs:
        print('process ' + di + '......')
        os.chdir(di)
        nfs = [x[2] for x in os.walk('nf/')][0]
        for nf in nfs:
            nf_im = Image.open('nf/' + nf).convert("L")
            hf_im = Image.open('hf/' + nf)
            try:
                im_lines, im_lines_words, lines_words, rotated = iu.crop_lines_words(nf_im)
                if rotated == True:
                    hf_im = hf_im.rotate(-90, expand=True)
                if len(lines_words) != 0:
                    im_labels = iu.formu_labels(hf_im, im_lines, im_lines_words)
                    lines_words = flat2d(lines_words)
                    lines_words = [x.resize((50, 50)) for x in lines_words]
                    im_labels = flat2d(im_labels)
                    if train == True:
                        lines_words, im_labels = under_sampling(lines_words, im_labels)
                else:
                    continue
            except Exception as e:
                print(di + nf)
                raise e
            images.extend(lines_words)
            labels.extend(im_labels)
            count += len(lines_words)
            
            if count > num * 10000:
                num += 1
                print("now count is:", count)

            if len(images) > 300000:
                os.chdir('../')
                writer = tf.python_io.TFRecordWriter(OUT_PATH + '-' + str(count // 300000))
                for i in range(len(images)):
                    image_raw = images[i].tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                                                'label': _init64_feature(labels[i]),
                                                'img': _bytes_feature(image_raw)}))
                    writer.write(example.SerializeToString())
                writer.close() 
                images = []
                labels = []
                os.chdir(di)
                
        
        os.chdir('../')
    
    writer = tf.python_io.TFRecordWriter(OUT_PATH + '-' + str(count // 300000 + 1))
    for i in range(len(images)):
        image_raw = images[i].tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                                   'label': _init64_feature(labels[i]),
                                   'img': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    print("the number of this tfrecord data:", count)
    os.chdir('../')

def under_sampling(words, labels):
    class0 = []
    class1 = []
    for i in range(len(labels)):
        if labels[i] == 0:
            class0.append((words[i], 0))
        elif labels[i] == 1:
            class1.append((words[i], 1))
    
    if len(class1) == 0:
        return words, labels
    np.random.shuffle(class0)
    class0 = class0[0:len(class1)]

    class0.extend(class1)
    np.random.shuffle(class0)
    words = [x[0] for x in class0]
    labels = [x[1] for x in class0]
    return words, labels

def over_sampling(words, labels):
    class0 = []
    class1 = []
    for i in range(len(labels)):
        if labels[i] == 0:
            class0.append((words[i], 0))
        elif labels[i] == 1:
            class1.append((words[i], 1))
    
    if len(class1) == 0:
        return words, labels
    cout = len(class0) // len(class1)
    class1_init = class1[:]
    for i in range(cout):
        class1.extend(class1_init)

    class0.extend(class1)
    np.random.shuffle(class0)
    words = [x[0] for x in class0]
    labels = [x[1] for x in class0]
    return words, labels

image_words_prep(IMG_DIR)
# image_words_prep(TEST_IMG_DIR, False)
