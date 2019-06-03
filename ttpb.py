# 命令行命令， 批量tex转换

import sys
import texf_topng
import os


def texf_topng_batch(filedir, den="200"):
    if not os.path.isdir(filedir):
        print("Need a dir with only tex files")
        return

    absfiledir = os.path.abspath(filedir)
    os.chdir(absfiledir)
    for root, dirs, files in os.walk(absfiledir):
        for name in files:
            try:
                texf_topng.tex_framed_topng(name, den)
            except Exception as e:
                print(e)


argn = len(sys.argv)
if argn == 1:
    print("Need a dir with only tex files.")
elif argn == 2:
    texf_topng_batch(sys.argv[1])
elif argn == 3:
    texf_topng_batch(sys.argv[1], sys.argv[2])
else:
    print("Too many arguments.")
