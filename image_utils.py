import PIL
from PIL import Image
import numpy as np


def crop_border(img, output_path, default_size=None):
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        if not default_size:
            old_im.save(output_path)
            return False
        else:
            assert len(default_size) == 2, default_size
            x_min, y_min, x_max, y_max = 0, 0, default_size[0], default_size[1]
            old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
            old_im.save(output_path)
            return False
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
    old_im.save(output_path)
    return True


def crop_lines(im):
    col, lin = im.size
    im_data = np.asarray(im)
    im_quarter_data = np.asarray(im.crop((0, 0, col // 4, lin)))
    im_reverse = np.where(im_data != 255, 1 / col, 0.)
    im_quarter_reverse = np.where(im_quarter_data != 255, 1 / (col // 4), 0.)

    im_sum = np.sum(im_reverse, axis=1)
    im_quarter_sum = np.sum(im_quarter_reverse, axis=1)

    lines_index_start = [0]
    lines_index_end = []
    for i in range(1, len(im_sum)):
        if im_sum[i] == 0 and im_sum[i-1] != 0:
            lines_index_end.append(i)
        elif im_sum[i] != 0 and im_sum[i-1] == 0:
            lines_index_start.append(i)
    lines_index_end.append(lin)

    im_lines = []
    for i in range(len(lines_index_start)):
        line_mid = (lines_index_end[i] + lines_index_start[i]) // 2
        line_height = lines_index_end[i] - lines_index_start[i]
        im_lines.append([lines_index_start[i], lines_index_end[i],
                         line_mid, line_height,
                         im_sum[line_mid], im_quarter_sum[line_mid]])
    i = 0
    lines = len(im_lines)

    heights = [im_lines[i][3] for i in range(lines)]
    height_mean = np.sum(heights) / lines
    while(i < lines):
        if im_lines[i][3] > 1.25 * height_mean or \
                im_lines[i][4] < 0.02 or \
                im_lines[i][5] < 0.02:
            del im_lines[i]
            lines -= 1
            continue
        i += 1

    for i in range(len(im_lines)):
        line_im = im.crop((0, im_lines[i][0], col, im_lines[i][1]))
        line_im.save("line-" + str(i) + ".png")

    return im_lines


def crop_words(im):
    col, lin = im.size
    im_data = np.asarray(im)
    im_reverse = np.where(im_data != 255, 1., 0.)
    im_sum = np.sum(im_reverse, axis=0)

    blank_start = []
    if im_sum[0] == 0:
        blank_start.append(0)
    blank_end = []
    for i in range(1, len(im_sum)):
        if im_sum[i] != 0 and im_sum[i-1] == 0:
            blank_end.append(i)
        elif im_sum[i] == 0 and im_sum[i-1] != 0:
            blank_start.append(i)
    if im_sum[-1] == 0:
        blank_end.append(col)

    im_blanks = []
    for i in range(len(word_start)):
        word_mid = (word_end[i] + word_start[i]) // 2
        word_width = word_end[i] - word_start[i]
        im_lines.append([lines_index_start[i], lines_index_end[i],
                         line_mid, line_height,
                         im_sum[line_mid], im_quarter_sum[line_mid]])
    return


def pad_goup_image(img, output_path, pad_size, buckets):
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = pad_size
    old_im = Image.open(img)
    old_size = (old_im.size[0]+PAD_LEFT+PAD_RIGHT,
                old_im.size[1]+PAD_TOP+PAD_BOTTOM)
    j = -1
    for i in range(len(buckets)):
        if old_size[0] <= buckets[i][0] and old_size[1] <= buckets[i][1]:
            j = i
            break
    if j < 0:
        new_size = old_size
        new_im = Image.new("RGB", new_size, (255, 255, 255))
        new_im.paste(old_im, (PAD_LEFT, PAD_TOP))
        new_im.save(output_path)
        return False
    new_size = buckets[j]
    new_im = Image.new("RGB", new_size, (255, 255, 255))
    new_im.paste(old_im, (PAD_LEFT, PAD_TOP))
    new_im.save(output_path)
    return True


def downsample_image(img, output_path, ratio):
    assert ratio >= 1, ratio
    if ratio == 1:
        return True
    old_im = Image.open(img)
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))

    new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
    new_im.save(output_path)
    return True
