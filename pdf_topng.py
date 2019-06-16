import PyPDF2
import PythonMagick
import sys

def to_png(name):
    pdfname = name

    pdf = open(pdfname, "rb")
    pdf_im = PyPDF2.PdfFileReader(pdf)
    npage = pdf_im.getNumPages()
    print("Converting %d pages..." % npage)
    for p in range(npage):
        im = PythonMagick.Image()
        im.density('200')
        im.read(pdfname + "[" + str(p) + "]")
        # im.defineValue("png", "bit-depth", "8")
        im.defineValue("png", "format", "png24")
        im.defineValue("png", "color-type", "2")
        print("    Converting %d/%d of %s..." % (p+1, npage, name))
        pngname = str(p + 1) + '.png'
        im.write(pngname)
    pdf.close()

    return npage

to_png(sys.argv[1])
