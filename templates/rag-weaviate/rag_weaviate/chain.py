import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
