import PIL
from PIL import Image
import numpy as np


def crop_border(img, default_size=None):
    im = Image.open(img)
    old_im = im.convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        if not default_size:
            old_im.save(output_path)
            return False
        else:
            assert len(default_size) == 2, default_size
            x_min, y_min, x_max, y_max = 0, 0, default_size[0], default_size[1]
            old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
            old_im.save(output_path)
            return False
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    out_im = im.crop((x_min, y_min, x_max + 1, y_max + 1))
    out_im.save(img)
    return (x_min, y_min)


def has_red_frame(img):
    im_data = np.asarray(Image.open(img))
    red = im_data[:, :, 0]
    green = im_data[:, :, 1]
    blue = im_data[:, :, 2]
    is_red = np.where(red == 255, 1, 0)
    not_green = np.where(green == 0, 1, 0)
    not_blue = np.where(blue == 0, 1, 0)
    has_red = is_red + not_blue + not_green
    red = np.sum(np.where(has_red == 3, 1, 0))
    if red == 0:
        return False
    else:
        return True


def formu_labels(img_hf, im_lines, im_lines_words):
    im_data = np.asarray(Image.open(img_hf))
    im_labels = []
    for i in range(len(im_lines)):
        im_labels.append([])
        print(im_lines[i][2])
        line = im_data[im_lines[i][2]]
        red = line[:, 0]
        green = line[:, 1]
        blue = line[:, 2]
        is_red = np.where(red == 255, 1, 0)
        not_green = np.where(green == 0, 1, 0)
        not_blue = np.where(blue == 0, 1, 0)
        has_red = is_red + not_blue + not_green
        red_inds = np.where(has_red == 3)[0]
        for j in range(len(im_lines_words[i])):
            for k in range(len(red_inds) // 2):
                if im_lines_words[i][j][0] > red_inds[k] and im_lines_words[i][
                        j][1] < red_inds[k + 1]:
                    im_labels[i].append(1)
                    break
            im_labels[i].append(0)

    return im_labels


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
        if im_sum[i] == 0 and im_sum[i - 1] != 0:
            lines_index_end.append(i)
        elif im_sum[i] != 0 and im_sum[i - 1] == 0:
            lines_index_start.append(i)
    lines_index_end.append(lin)

    im_lines = []
    for i in range(len(lines_index_start)):
        line_mid = (lines_index_end[i] + lines_index_start[i]) // 2
        line_height = lines_index_end[i] - lines_index_start[i]
        im_lines.append([
            lines_index_start[i], lines_index_end[i], line_mid, line_height,
            im_sum[line_mid], im_quarter_sum[line_mid]
        ])
    i = 0
    lines = len(im_lines)

    heights = [im_lines[i][3] for i in range(lines)]
    height_down = np.percentile(heights, 10)
    height_up = np.percentile(heights, 90)
    while (i < lines):
        if im_lines[i][3] > height_up * 1.25 or \
                im_lines[i][3] < height_down * 0.5 or \
                im_lines[i][4] < 0.02 or \
                im_lines[i][5] < 0.02:
            del im_lines[i]
            lines -= 1
            continue
        i += 1

    lines_words = []
    im_lines_words = []
    for i in range(len(im_lines)):
        line_im = im.crop((0, im_lines[i][0], col, im_lines[i][1]))
        im_words, words = crop_words(line_im)
        im_lines_words.append(im_words)
        lines_words.append(words)

    im_lines = [x[0:3] for x in im_lines]

    return im_lines, im_lines_words, lines_words


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
        if im_sum[i] != 0 and im_sum[i - 1] == 0:
            blank_end.append(i)
        elif im_sum[i] == 0 and im_sum[i - 1] != 0:
            blank_start.append(i)
    if im_sum[-1] == 0:
        blank_end.append(col)

    im_blanks = []
    for i in range(len(blank_start)):
        blank_mid = (blank_start[i] + blank_end[i]) // 2
        blank_width = blank_end[i] - blank_start[i]
        im_blanks.append([blank_start[i], blank_end[i], blank_width])

    # 找出非字母间隔的空白
    wids = np.asarray([im_blanks[i][2] for i in range(len(im_blanks))])
    up_wid = np.percentile(wids, 95)
    wid_inds = np.where(wids < up_wid)[0]
    wid_mean = 0
    for i in range(len(wid_inds)):
        wid_mean += wids[wid_inds[i]]
    wid_mean = wid_mean / len(wid_inds)
    is_blank = np.where(wids > wid_mean, 1, 0)

    i = 0
    up = len(im_blanks)
    is_blank = list(is_blank)
    while i < up:
        if is_blank[i] == 0:
            del im_blanks[i]
            del is_blank[i]
            up -= 1
        else:
            i += 1

    im_words = []
    if not im_blanks[0][0] == 0:
        im_words.append([0, im_blanks[0][0]])
    for i in range(len(im_blanks) - 1):
        im_words.append([im_blanks[i][1], im_blanks[i + 1][0]])
    if im_blanks[-1][1] != col:
        im_words.append([im_blanks[-1][1], col])

    words = []
    for i in range(len(im_words)):
        word_im = im.crop((im_words[i][0], 0, im_words[i][1], lin))
        words.append(word_im)

    return im_words, words


def pad_goup_image(img, output_path, pad_size, buckets):
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = pad_size
    old_im = Image.open(img)
    old_size = (old_im.size[0] + PAD_LEFT + PAD_RIGHT,
                old_im.size[1] + PAD_TOP + PAD_BOTTOM)
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
    new_size = (int(old_size[0] / ratio), int(old_size[1] / ratio))

    new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
    new_im.save(output_path)
    return True
