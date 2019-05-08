import re
import os
import PyPDF2
import PythonMagick
from pdflatex import PDFLaTeX


def tex_framed_tojpg(filen, den="200"):
    if not os.path.isfile(filen):
        print(os.getcwd())
        print("%s is not a file name." % filen)
        return

    filedir = os.path.dirname(filen)
    filename = os.path.basename(filen)

    if filedir != '':
        os.chdir(filedir)

    with open(filename, mode='r') as fl:
        try:
            text = fl.read()
        except Exception as e:
            print(e)
            return

    nam_suf = re.search(r"\.", filename)
    # nf is 'no frame'
    if nam_suf:
        nam_pos = nam_suf.span()
        filename_nosuf = filename[:nam_pos[0]]
        filename_nf = filename[:nam_pos[0]] + "_nf" + filename[nam_pos[0]:]
    else:
        filename_nosuf = filename
        filename_nf = filename + "_nf"

    # add package xcolor
    # deal with 'documentstyle'
    text = re.sub(r"\\documentstyle", "\\documentclass", text)
    text = re.sub(r"(?<!%)\\usepackage.*?\{epsfig}", "", text)
    if not re.search(r"(?<!%)\\usepackage.*?\{xcolor}", text):
        pack = re.search(r"(?<!%)\\documentclass.*?\{\w*?}", text)
        if pack:
            pack_pos = pack.span()
            text = text[:pack_pos[1]] \
                + "\n\\usepackage[svgnames]{xcolor}\n" \
                + text[pack_pos[1]:]

    text1 = text
    text2 = text
    write_nf = False
    pattern = r'(?<!\\fcolorbox\{Red}  \
        \{White}\{)(?<!\$|\\)\$(?!\$|\}).*?(?<!\\)\$'
    formu = re.search(pattern, text2, re.S)
    if formu:
        write_nf = True
    while formu:
        pos = formu.span()
        text = text[:pos[0]] \
            + "\\fcolorbox{Red}  {White}{" \
            + text[pos[0]: pos[1]] + "}" + text[pos[1]:]
        text1 = text1[:pos[0]] \
            + "\\fcolorbox{White}{White}{" \
            + text1[pos[0]: pos[1]] + "}" + text1[pos[1]:]
        text2 = text2[:pos[0]] \
            + "\\fcolorbox{Red}  {White}{ " \
            + text2[pos[0] + 1: pos[1] - 1] + " }" \
            + text2[pos[1]:]
        formu = re.search(pattern, text2, re.S)

    with open(filename, mode='w') as fl:
        fl.write(text)

    if write_nf:
        with open(filename_nf, mode='w') as fl_nf:
            fl_nf.write(text1)

    # get pdfs
    pdfl = PDFLaTeX.from_texfile(filename)
    pdfl_nf = PDFLaTeX.from_texfile(filename_nf)
    try:
        pdfl.create_pdf(keep_pdf_file=True)
        pdfl_nf.create_pdf(keep_pdf_file=True)
    except FileNotFoundError:
        print("Lack dependencies to get pdfs of %s." % filename)
        return
    except Exception as e:
        print(e)

    if not os.path.isdir(filename_nosuf + "_images"):
        os.mkdir(filename_nosuf + "_images")
    os.chdir(filename_nosuf + "_images")

    for pdffilename in [chsuf(filename, "pdf"), chsuf(filename_nf, "pdf")]:
        pdf_im = PyPDF2.PdfFileReader(open("..//" + pdffilename[0], "rb"))
        npage = pdf_im.getNumPages()
        print("Converting %d pages..." % npage)
        for p in range(npage):
            im = PythonMagick.Image()
            im.density(den)
            im.read("..//" + pdffilename[0] + '[' + str(p) + ']')
            # im.defineValue("png", "bit-depth", "8")
            im.defineValue("png", "format", "png24")
            im.defineValue("png", "color-type", "2")
            print("    Converting %d/%d of %s..."
                  % (p+1, npage, pdffilename[1]))
            im.write(pdffilename[1] + "-" + str(p + 1) + '.png')

    # os.remove(filename_nf)
    # os.remove(chsuf(filename, "pdf"))
    # os.remove(chsuf(filename_nf, "pdf"))
    os.chdir("..")


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
