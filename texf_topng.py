import re
import os
import PyPDF2
import PythonMagick
from pdflatex import PDFLaTeX
import image_utils as iu


def tex_framed_topng(filen, den="200"):
    if not os.path.isfile(filen):
        print(os.getcwd())
        print("%s is not a file name." % filen)
        return

    # filen: name with dir and ext
    # filename: name with ext
    # flname: name but no ext
    ROOT = os.path.abspath('./')
    filedir, filename = os.path.split(filen)
    flname, ext = os.path.splitext(filename)
    if filedir != '':
        os.chdir(filedir)

    # nf is 'no frame'
    flname_nf = flname + "_nf"

    with open(flname, mode='r') as fl:
        try:
            text = fl.read()
        except Exception as e:
            print(e)
            return

    # deal with text
    text, text1, write_nf = text_treatment(text)

    with open(flname, mode='w') as fl:
        fl.write(text)

    if write_nf:
        with open(flname_nf, mode='w') as fl_nf:
            fl_nf.write(text1)

    pdfl = PDFLaTeX.from_texfile(flname)
    pdfl_nf = PDFLaTeX.from_texfile(flname_nf)
    try:
        pdfl.create_pdf(keep_pdf_file=True)
        pdfl_nf.create_pdf(keep_pdf_file=True)
    except Exception as e:
        print(e)
        return

    if not os.path.isdir(flname + "_images"):
        os.mkdir(flname + "_images")
        os.mkdir(flname + "_images/nf")
        os.mkdir(flname + "_images/hf")

    os.chdir(flname + "_images/hf")
    no_frame_pngs = to_png(flname, den)
    os.chdir("../nf")
    to_png(flname_nf, den)
    os.chdir('../')

    for png in no_frame_pngs:
        os.remove('hf/' + png)
        os.remove('nf/' + png)

    os.chdir(ROOT)


# generate pdf and convert into png
def to_png(name, den):
    pdfname = "../../" + name + ".pdf"
    no_frame_pngs = []

    pdf = open(pdfname, "rb")
    pdf_im = PyPDF2.PdfFileReader(pdf)
    npage = pdf_im.getNumPages()
    print("Converting %d pages..." % npage)
    for p in range(npage):
        im = PythonMagick.Image()
        im.density(den)
        im.read(pdfname + "[" + str(p) + "]")
        # im.defineValue("png", "bit-depth", "8")
        im.defineValue("png", "format", "png24")
        im.defineValue("png", "color-type", "2")
        print("    Converting %d/%d of %s..." % (p+1, npage, name))
        pngname = str(p + 1) + '.png'
        im.write(pngname)
        iu.crop_border(pngname)
        if not iu.has_red_frame(pngname):
            no_frame_pngs.append(pngname)

    pdf.close()

    return no_frame_pngs


def text_treatment(text):
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

    return text, text1, write_nf
