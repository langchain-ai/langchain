import os

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate

if os.environ.get("WEAVIATE_API_KEY", None) is None:
    raise Exception("Missing `WEAVIATE_API_KEY` environment variable.")

if os.environ.get("WEAVIATE_ENVIRONMENT", None) is None:
    raise Exception("Missing `WEAVIATE_ENVIRONMENT` environment variable.")

WEAVIATE_INDEX_NAME = os.environ.get("WEAVIATE_INDEX", "langchain-test")

### Ingest code - you may need to run this the first time
# Load
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# # Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# # Add to vectorDB
# vectorstore = Weaviate.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings(), index_name=WEAVIATE_INDEX_NAME
# )
# retriever = vectorstore.as_retriever()

vectorstore = Weaviate.from_existing_index(WEAVIATE_INDEX_NAME, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI()
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)
