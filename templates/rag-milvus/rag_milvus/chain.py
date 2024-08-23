from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_milvus.vectorstores import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Example for document loading (from url), splitting, and creating vectorstore

# Setting the URI as a local file, e.g.`./milvus.db`, is the most convenient method,
# as it automatically utilizes Milvus Lite to store all data in this file.
#
# If you have large scale of data such as more than a million docs,
# we recommend setting up a more performant Milvus server on docker or kubernetes.
# (https://milvus.io/docs/quickstart.md)
# When using this setup, please use the server URI,
# e.g.`http://localhost:19530`, as your URI.

URI = "./milvus.db"

""" 
# Load
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Milvus.from_documents(documents=all_splits,
                                    collection_name="rag_milvus",
                                    embedding=OpenAIEmbeddings(),
                                    drop_old=True,
                                    connection_args={"uri": URI},
                                    )
retriever = vectorstore.as_retriever()
"""

# Embed a single document as a test
vectorstore = Milvus.from_texts(
    ["harrison worked at kensho"],
    collection_name="rag_milvus",
    embedding=OpenAIEmbeddings(),
    drop_old=True,
    connection_args={"uri": URI},
)
retriever = vectorstore.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI()

# RAG chain
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
