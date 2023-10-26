import os 
import pinecone
from operator import itemgetter
from langchain.vectorstores import Pinecone
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel

# Pinecone init
# Find API key in console at app.pinecone.io
YOUR_API_KEY = os.getenv('PINECONE_API_KEY') or 'PINECONE_API_KEY'
# Find ENV (cloud region) next to API key in console
YOUR_ENV = os.getenv('PINECONE_ENVIRONMENT') or 'PINECONE_ENV'
# Init
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)

# Get vectorstore
text_field = "text"
index_name = "langchain-multi-query-demo"
index = pinecone.Index(index_name)
vectorstore = Pinecone(index, 
                       OpenAIEmbeddings(), 
                       text_field)

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
