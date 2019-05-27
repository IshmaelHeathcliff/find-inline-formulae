import tensorflow as tf
import numpy as np
import sys
import os
import image_utils as iu
from PIL import Image

IMG_DIR = '2003/Images/train'
TEST_IMG_DIR = 'test/Images'
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


def image_words_prep(IMG_DIR):
    os.chdir(IMG_DIR)

    dirs = [x[1] for x in os.walk('./')][0]
    images = []
    labels = []
    heights = []
    widths = []
    for di in dirs:
        print('process ' + di + '......')
        os.chdir(di)
        nfs = [x[2] for x in os.walk('nf/')][0]
        for nf in nfs:
            nf_im = Image.open('nf/' + nf).convert("L")
            try:
                im_lines, im_lines_words, lines_words = iu.crop_lines_words(nf_im)
                im_labels = iu.formu_labels('hf/' + nf, im_lines, im_lines_words)
            except Exception as e:
                print(di + nf)
                raise e
            lines_words = flat2d(lines_words)
            im_labels = flat2d(im_labels)
            height = [x.size[1] for x in lines_words]
            width = [x.size[0] for x in lines_words]
            images.extend(lines_words)
            labels.extend(im_labels)
            heights.extend(height)
            widths.extend(width)
    
    writer = tf.python_io.TFRecordWriter(OUT_PATH)
    for i in range(len(images)):
        image_raw = images[i].tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                                   'label': _init64_feature(labels[i]),
                                   'height': _init64_feature(heights[i]),
                                   'width': _init64_feature(widths[i]),
                                   'img': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close() 
    os.chdir('../')


image_words_prep(TEST_IMG_DIR)
