import numpy as np
import sys
import os
import image_utils as iu
from PIL import Image


def image_words_prep(imgpath):
    nfs = [x[2] for x in os.walk('nf/')][0]

    data = []
    for nf in nfs:
        nf_im = Image.open('nf/' + nf).convert("L")
        im_lines, im_lines_words, lines_words = iu.crop_lines(nf_im)
        im_labels = iu.formu_labels('hf/' + nf, im_lines, im_lines_words)
        data.append([im_lines, im_lines_words, lines_words, im_labels])

    return np.asarray(data)


ROOT = os.path.abspath('./')
imgpath = sys.argv[1]
os.chdir(imgpath)
img_data = image_words_prep(imgpath)
np.save(os.path.basename(imgpath), img_data)
os.chdir(ROOT)
