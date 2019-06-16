#coding=utf-8
# 图片处理工具合集

import PIL
from PIL import Image
import numpy as np
import math


def crop_border(img, save=True): # 去除空白边框
    im = Image.open(img)
    old_im = im.convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        return (0, 0)
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    out_im = im.crop((x_min, y_min, x_max + 1, y_max + 1))
    if save == False:
        return out_im, (x_min, y_min)
    out_im.save(img)
    return


def has_red_frame(img): # 判断是否含有红框
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


def formu_labels(hf_im, im_lines, im_lines_words): # 获得标记信息
    im_data = np.asarray(hf_im)
    im_labels = []
    for i in range(len(im_lines)):
        im_labels.append([])
        line = im_data[im_lines[i][2]]
        red = line[:, 0]
        green = line[:, 1]
        blue = line[:, 2]
        is_red = np.where(red == 255, 1, 0)
        not_green = np.where(green < 255, 1, 0)
        not_blue = np.where(blue < 255, 1, 0)
        has_red = is_red + not_blue + not_green
        red_inds = []
        for k in range(len(has_red) - 1):
            if has_red[k] == 3 and has_red[k+1] != 3:
                red_inds.append(k)
        if has_red[-1] == 3 and has_red[-2] != 3:
            red_inds.append(len(has_red) - 1)

        for j in range(len(im_lines_words[i])):
            cont = False
            for k in range(len(red_inds) // 2):
                if im_lines_words[i][j][0] > red_inds[2*k] and im_lines_words[i][j][1] < red_inds[2*k + 1]:
                    im_labels[i].append(1)
                    cont = True
                    break
            if cont == True:
                continue
            else:
                im_labels[i].append(0)

    return im_labels


def crop_lines(im, save_line_im=False, check_rotate=True): # 获得文行信息
    col, lin = im.size
    im_data = np.asarray(im)
    im_reverse = np.where(im_data != 255, 1., 0.)

    # 处理图像方向
    rotated = False
    im_sum_x = np.sum(im_reverse, axis=1)
    im_sum_y = np.sum(im_reverse, axis=0)
    im_blank_x = np.sum(np.where(im_sum_x == 0, 1, 0)) / lin
    im_blank_y = np.sum(np.where(im_sum_y == 0, 1, 0)) / col
    if im_blank_y > im_blank_x and check_rotate == True:
        im_data = np.rot90(im_data, -1)
        im = Image.fromarray(im_data)
        im_sum = im_sum_y / lin
        lin, col = im_data.shape
        rotated = True
    else:
        im_sum = im_sum_x / col

    im_quarter_data = im_data[:, 0:(col // 4)]
    im_quarter_reverse = np.where(im_quarter_data != 255, 1 / (col // 4), 0.)
    im_quarter_sum = np.sum(im_quarter_reverse, axis=1)

    # 每行位置
    lines_index_start = [0]
    lines_index_end = []
    for i in range(1, len(im_sum)):
        if im_sum[i] == 0 and im_sum[i - 1] != 0:
            lines_index_end.append(i)
        elif im_sum[i] != 0 and im_sum[i - 1] == 0:
            lines_index_start.append(i)
    lines_index_end.append(lin)

    im_lines = [] # [[start, end, mid, height, sum, quater_sum, start_sum, end_sum], ...]
    for i in range(len(lines_index_start)):
        line_start = lines_index_start[i]
        line_end = lines_index_end[i]
        line_mid = (line_start + line_end) // 2
        line_height = line_end - line_start
        line_sum = np.mean(im_sum[line_start:line_end])
        line_quater_sum = np.mean(im_quarter_sum[line_start:line_end])
        im_lines.append([
            lines_index_start[i], 
            lines_index_end[i], 
            line_mid, 
            line_height,
            line_sum, 
            line_quater_sum, 
            im_sum[line_start], 
            im_sum[line_end - 1]
        ])
    i = 0
    lines = len(im_lines)

    # 筛选
    heights = [im_lines[i][3] for i in range(lines)]
    height_mean, std = min_square(heights)
    # print(heights, height_mean, std)
    if std < 0.25 * height_mean:
        height_mean = height_mean * 1.5
    else:
        height_mean += std
    height_down = np.percentile(heights, 10)
    # print([im_lines[i][5] for i in range(lines)])
    # print([im_lines[i][4] for i in range(lines)])

    # 筛选文行，造训练数据时严格，实际使用可放松
    while (i < lines):
        if (im_lines[i][3] > height_mean and im_lines[i][5] < 0.13) or \
                (im_lines[i][5] < 0.1 and im_lines[i][4] < 0.13) or \
                im_lines[i][3] < height_down * 0.9 or \
                im_lines[i][4] > 0.8 or \
                im_lines[i][6] > 0.8 or \
                im_lines[i][7] > 0.8 or \
                im_lines[i][5] > 0.8:
            del im_lines[i]
            lines -= 1
            continue
        i += 1

    if save_line_im:
        for i in range(len(im_lines)):
            line_im = im.crop((0, im_lines[i][0], col, im_lines[i][1]))
            line_im.save('line-' + str(i) + '.png')

    return im_lines, rotated


def crop_lines_words(im, check_rotate=True): # 获得整张图的单词信息及单词图片
    col, lin = im.size
    lines_words = []
    im_lines_words = []
    im_lines, rotated = crop_lines(im, False, check_rotate)
    for i in range(len(im_lines)):
        line_im = im.crop((0, im_lines[i][0], col, im_lines[i][1]))
        im_words, words = crop_words(line_im)
        im_lines_words.append(im_words)
        lines_words.append(words)

    im_lines = [x[0:3] for x in im_lines] # [[start, end, mid], ...]

    return im_lines, im_lines_words, lines_words, rotated


def crop_words(im, save_words=False): # 获得单行单词信息及图片
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
    if len(im_blanks) == 0:
        return [[0, col]], [im]
    wids = [im_blanks[i][2] for i in range(len(im_blanks))]
    # 将最大的两个空白宽度改为次大
    max_wid1 = max(wids)
    max_ind1 = wids.index(max_wid1)
    wids[max_ind1] = 0
    max_wid2 = max(wids)
    max_ind2 = wids.index(max_wid2)
    wids[max_ind2] = 0
    
    inf_max_wid = max(wids)
    wids[max_ind1] = inf_max_wid
    wids[max_ind2] = inf_max_wid
    wids = np.asarray(wids)

    wid_mean = 0
    if len(wids) != 0:
        wid_mean, std = min_square(wids)
    # wid_mean = wid_mean + 0.2 * std
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

    im_words = [] # [[start, end], ...]

    if len(im_blanks) == 0:
        return [[0, col]], [im]
    try:
        if im_blanks[0][0] != 0:
            im_words.append([0, im_blanks[0][0]])
    except Exception as e:
        print(im_blanks)
        im.save('error.png')
        raise e
    for i in range(len(im_blanks) - 1):
        im_words.append([im_blanks[i][1], im_blanks[i + 1][0]])
    if im_blanks[-1][1] != col:
        im_words.append([im_blanks[-1][1], col])

# ========================================================
# 将长单词切割为多个短图片
    # i = 0
    # up = len(im_words)
    # while i < up:
    #     if im_words[i][1] - im_words[i][0] >= 2 * lin:
    #         im_words.append([im_words[i][0], im_words[i][0] + lin])
    #         im_words.append([im_words[i][0] + lin, im_words[i][1]])
    #         del im_words[i]
    #         up += 1
    #     else:
    #         i +=1

    # im_words.sort()
# =============================================================
    
    # i = 0
    # up = len(im_words)
    # while i < up:
    #     if im_words[i][1] - im_words[i][0] < lin * 0.3:
    #         del im_words[i]
    #         up -= 1
    #     else:
    #         i += 1

    words = []
    for i in range(len(im_words)):
        word_im = im.crop((im_words[i][0], 0, im_words[i][1], lin))
        if save_words == True:
            word_im.save("word-" + str(i) + ".png")
        words.append(word_im)

    return im_words, words


def min_square(lis): # 最小二乘法
    num = len(lis)
    lis = np.asarray(lis)
    mi, ma = np.min(lis), np.max(lis)
    out_lis = []
    if mi == ma:
        return lis, 0

    li = mi
    for i in range(ma - mi):
        sq = np.sum((lis - li)**2)
        out_lis.append(math.sqrt(sq / num))
        li += 1
    out = out_lis.index(min(out_lis))
    return mi + out, out_lis[out]

