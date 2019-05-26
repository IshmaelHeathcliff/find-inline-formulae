import numpy as np
import sys
import os
import image_utils as iu
from PIL import Image

IMG_DIR = '2003/Images'
TEST_IMG_DIR = 'test/Images'


def image_words_prep(IMG_DIR):
    os.chdir(IMG_DIR)

    dirs = [x[1] for x in os.walk('./')][0]
    data = []
    for di in dirs:
        print('process ' + di + '......')
        os.chdir(di)
        nfs = [x[2] for x in os.walk('nf/')][0]
        for nf in nfs:
            nf_im = Image.open('nf/' + nf).convert("L")
            try:
                im_lines, im_lines_words, lines_words = iu.crop_lines_words(
                    nf_im)
                im_labels = iu.formu_labels('hf/' + nf, im_lines,
                                            im_lines_words)
            except Exception as e:
                print(di + nf)
                raise e
            

            data.append([im_lines, im_lines_words, lines_words, im_labels])
        os.chdir('../')

    return np.asarray(data)


img_data = image_words_prep(TEST_IMG_DIR)
np.save(os.path.basename(IMG_DIR), img_data)
