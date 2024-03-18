import io
import os
import numpy as np

from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def pdf_loader(file_path):
    try:
        import fitz  # noqa:F401
        import easyocr
    except ImportError:
        raise ImportError(
            "`PyMuPDF` or 'easyocr' package is not found, please install it with "
            "`pip install pymupdf or pip install easyocr.`"
        )
    
    doc = fitz.open(file_path)
    reader = easyocr.Reader(['en'])
    result =''
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pagetext = page.get_text().strip()
        if pagetext:
            result=result+pagetext
        if len(doc.get_page_images(i)) > 0 :
            for img in doc.get_page_images(i):
                if img:
                    pageimg=''
                    xref = img[0]
                    img_data = doc.extract_image(xref)
                    img_bytes = img_data['image']
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    img = np.array(pil_image)
                    img_result = reader.readtext(img, paragraph=True, detail=0)
                    pageimg=pageimg + ', '.join(img_result).strip()
                    if pageimg.endswith('!') or pageimg.endswith('?') or pageimg.endswith('.'):
                        pass
                    else:
                        pageimg=pageimg+'.'
                result=result+pageimg
    return result

def ingest_documents():
    """
    Ingest PDF to Redis from the data/ directory that
    contains Edgar 10k filings data for Nike.
    """
    # Load list of pdfs
    data_path = "data/"
    doc_path = [os.path.join(data_path, file) for file in os.listdir(data_path)][0]

    print("Parsing 10k filing doc for NIKE", doc_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, add_start_index=True
    )
    content = pdf_loader(doc_path)
    chunks = text_splitter.split_text(content)
    # loader = UnstructuredFileLoader(doc, mode="single", strategy="fast")
    # chunks = loader.load_and_split(text_splitter)

    print("Done preprocessing. Created ", len(chunks), " chunks of the original pdf")

    # Create vectorstore
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = []
    for chunk in chunks:
        doc = Document(page_content=chunk)
        documents.append(doc)

    # Add to vectorDB
    _ = Chroma.from_documents(
        documents=documents,
        collection_name="gaudio-rag",
        embedding=embedder,
        persist_directory='/tmp/gaudi_rag_db'
    )


if __name__ == "__main__":
    ingest_documents()