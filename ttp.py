# 命令行命令， 单个tex转换

import sys
import texf_topng

argn = len(sys.argv)
if argn == 1:
    print("Need a tex file name.")
elif argn == 2:
    texf_topng.tex_framed_topng(sys.argv[1])
elif argn == 3:
    texf_topng.tex_framed_topng(sys.argv[1], sys.argv[2])
else:
    print("Too much arguments.")
