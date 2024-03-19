import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SingleStoreDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

if os.environ.get("SINGLESTOREDB_URL", None) is None:
    raise Exception("Missing `SINGLESTOREDB_URL` environment variable.")

# SINGLESTOREDB_URL takes the form of: "admin:password@host:port/db_name"

## Ingest code - you may need to run this the first time
# # Load
# from langchain_community.document_loaders import WebBaseLoader

# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# # Split
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# # Add to vectorDB
# vectorstore = SingleStoreDB.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings()
# )
# retriever = vectorstore.as_retriever()

vectorstore = SingleStoreDB(embedding=OpenAIEmbeddings())
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


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
