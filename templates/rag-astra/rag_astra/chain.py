import os
import cassio

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Cassandra

if os.environ.get("ASTRA_DB_APPLICATION_TOKEN") is None:
    raise Exception("Missing `ASTRA_DB_APPLICATION_TOKEN` environment variable.")

if os.environ.get("ASTRA_DB_ID") is None:
    raise Exception("Missing `ASTRA_DB_ID` environment variable.")

ASTRA_DB_COLLECTION_NAME = os.environ.get("ASTRA_DB_COLLECTION_NAME", "rag_astra")

# Use cassio to initialize the Astra DB connection
cassio.init(
    token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
    database_id=os.environ.get("ASTRA_DB_ID"),
)

# OPTION 1: Example code for ingesting documents into Astra DB - adapt as needed
""" 
# Load the documents
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split the text
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to Astra DB
vectorstore = Cassandra.from_documents(
    documents=all_splits, 
    collection_name="rag-astra",
    embedding=OpenAIEmbeddings(),
)

# Create the retriever object
retriever = vectorstore.as_retriever()
"""

# Option 2: Initialize the Vector Store with some text
vectorstore = Cassandra.from_texts(
    [
        "Astra DB gives developers the APIs, real-time data and complete ecosystem integrations to put accurate RAG and Gen AI apps in production - FAST."
    ],
    session=None,
    keyspace=None,
    embedding=OpenAIEmbeddings(),
    table_name=ASTRA_DB_COLLECTION_NAME,
)
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
