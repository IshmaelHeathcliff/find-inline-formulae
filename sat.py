import re
import os
import sys
import PyPDF2
import PythonMagick
from pdflatex import PDFLaTeX

def sat(test, den="200"):
    with open(test, mode='r') as fl:
        text = fl.read()

    nam_suf = re.search(r"\.", test)
    # nf is 'no frame'
    if nam_suf:
        nam_pos = nam_suf.span()
        test_nosuf = test[:nam_pos[0]]
        test_nf = test[:nam_pos[0]] + "_nf" + test[nam_pos[0]:]
    else:
        test_nosuf = test
        test_nf = test + "_nf"

    # add package xcolor
    text = re.sub(r"\\documentstyle", "\\documentclass", text) # deal with 'documentstyle'
    if not re.search(r"(?<!%)\\usepackage.*?\{xcolor}", text):
        pack = re.search(r"(?<!%)\\documentclass.*?\{\w*?}", text)
        if pack:
            pack_pos = pack.span()
            text = text[:pack_pos[1]] + "\n\\usepackage[svgnames]{xcolor}\n" + text[pack_pos[1]:]

    text1 = text
    text2 = text
    write_nf = False
    pattern = r'(?<!\\fcolorbox\{Red}  \{White}\{)(?<!\$|\\)\$(?!\$|\}).*?(?<!\\)\$'
    formu =  re.search(pattern, text2, re.S)
    if formu:
        write_nf = True
    while formu:
        pos = formu.span()
        text = text[:pos[0]] + "\\fcolorbox{Red}  {White}{" + text[pos[0]: pos[1]] + "}" + text[pos[1]:]
        text1 = text1[:pos[0]] + "\\fcolorbox{White}{White}{" + text1[pos[0]: pos[1]] + "}" + text1[pos[1]:]
        text2 = text2[:pos[0]] + "\\fcolorbox{Red}  {White}{ " + text2[pos[0] + 1: pos[1] - 1] + " }" + text2[pos[1]:]
        formu =  re.search(pattern, text2, re.S)


    with open(test, mode='w') as fl:
        fl.write(text)

    if write_nf:
        with open(test_nf, mode='w') as fl_nf:
            fl_nf.write(text1)
        
    # get pdfs
    pdfl = PDFLaTeX.from_texfile(test)
    pdfl_nf = PDFLaTeX.from_texfile(test_nf)
    try:
        pdfl.create_pdf(keep_pdf_file=True)
        pdfl_nf.create_pdf(keep_pdf_file=True)
    except FileNotFoundError:
        print("failed to get pdfs of %s." % test)
        return

    if not os.path.isdir(test_nosuf + "_images"):
        os.mkdir(test_nosuf + "_images")
    os.chdir(test_nosuf + "_images")
    

    for pdffilename in [chsuf(test, "pdf"), chsuf(test_nf, "pdf")]:
        pdf_im = PyPDF2.PdfFileReader(open("..//" + pdffilename[0], "rb"))
        npage = pdf_im.getNumPages() 
        print("Converting %d pages..." % npage)
        for p in range(npage):
            im = PythonMagick.Image()
            im.density(den)
            im.magick = "RGB"
            im.read("..//" + pdffilename[0] + '[' + str(p) +']')
            print("    Converting %d/%d of %s..." % (p+1, npage, pdffilename[1]))
            im.write(pdffilename[1] + "-" + str(p)+ '.jpg')

def chsuf(filename, suf):
    nam_suf = re.search(r"\.", filename)
    if nam_suf:
        nam_pos = nam_suf.span()
        outname = filename[:nam_pos[0]] + "." + suf
        flname = filename[:nam_pos[0]]
    else:
        outname = filename + "." + suf
        flname = filename

    return outname, flname

argn = len(sys.argv)
if argn == 1:
    print("Need a tex file name.")
elif argn == 2:
    sat(sys.argv[1])
elif argn == 3:
    sat(sys.argv[1], sys.argv[2])
else:
    print("Too much arguments.")
