import sys
import texf_tojpg

argn = len(sys.argv)
if argn == 1:
    print("Need a tex file name.")
elif argn == 2:
    try:
        texf_tojpg.tex_framed_tojpg(sys.argv[1])
    except Exception as e:
        print(e)
elif argn == 3:
    try:
        texf_tojpg.tex_framed_tojpg(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(e)
else:
    print("Too much arguments.")
