from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from .uniquechunkretriever import UniqueChunkRetriever

docs = []

for pdf in pdfs_list:
    loader = PyPDFLoader(pdf)
    sub_docs = loader.load()
    for doc in sub_docs:
        doc.metadata["source"] = pdf
    docs.extend(sub_docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

for id, chunk in enumerate(chunks):
    chunk.metadata["UID"] = id
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
