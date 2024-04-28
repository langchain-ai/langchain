from langchain_experimental.unique_chunks_retriever import UniqueChunkRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# 1) Load vector store from a pdf file

docs = []
loader = PyPDFLoader("World_Bank.pdf")
sub_docs = loader.load()
docs.extend(sub_docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# important! each chunk needs an UID in its metadata
for id, chunk in enumerate(chunks):
    chunk.metadata["UID"] = id
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 2) Use the UniqueChunkRetriever to get unique chunks for each query

ucr = UniqueChunkRetriever(vectorstore)
queries = ["Who published this document?", "What is the economic growth?"]
# these are the most relevant chunks for each query,
# all returned chunks are unique and do not overlap with other retrieved chunks for other queries
three_unique_chunks_for_each_query = ucr.optimize(queries_list=queries, k=3)
