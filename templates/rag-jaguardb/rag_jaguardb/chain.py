import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.jaguar import Jaguar
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)

if os.environ.get("JAGUAR_API_KEY", None) is None:
    raise Exception("Missing `JAGUAR_API_KEY` environment variable.")
JAGUAR_API_KEY = os.environ["JAGUAR_API_KEY"]

url = "http://192.168.3.88:8080/fwww/"
pod = "vdb"
store = "langchain_test_store"
vector_index = "v"
vector_type = "cosine_fraction_float"
vector_dimension = 1536
embeddings = OpenAIEmbeddings()
vectorstore = Jaguar(
    pod, store, vector_index, vector_type, vector_dimension, url, embeddings
)

retriever = vectorstore.as_retriever()

vectorstore.login()
"""
Create vector store on the JaguarDB database server.
This should be done only once.
"""

metadata = "category char(16)"
text_size = 4096
vectorstore.create(metadata, text_size)

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# RAG
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
