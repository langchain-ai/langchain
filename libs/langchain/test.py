from langchain.document_loaders.pdf import *

def test():
    loader = PDFMinerLoader("test.pdf")
    # loader = PyMuPDFLoader("test.pdf", extract_images = True)
    doc = loader.load()

    print(doc)

test()