import getpass
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_nvidia_aiplay import ChatNVIDIA, NVIDIAEmbeddings
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langserve.client import RemoteRunnable

EMBEDDING_MODEL = "nvolveqa_40k"
CHAT_MODEL = "gpt-43b-905"
HOST = "127.0.0.1"
PORT = "19530"
COLLECTION_NAME = "test"
os.getenv('NGC_API_KEY')

# Read from Milvus Vector Store
embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Milvus(
    connection_args={"host": HOST, "port": PORT},
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever()

# RAG prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
prompt = ChatPromptTemplate.from_template(template)


# Guardrails
config = RailsConfig.from_path("./guardrails/config")
guardrails = RunnableRails(config, input_key="question", output_key="answer", passthrough=False)

# RAG
model = ChatNVIDIA(model=CHAT_MODEL)

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
chain_with_guardrails = guardrails | chain
print(chain_with_guardrails.invoke({"question": "How many Americans receive Social Security Benefits?"}))