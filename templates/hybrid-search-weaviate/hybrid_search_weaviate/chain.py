import os

import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Check env vars
if os.environ.get("WEAVIATE_API_KEY", None) is None:
    raise Exception("Missing `WEAVIATE_API_KEY` environment variable.")

if os.environ.get("WEAVIATE_ENVIRONMENT", None) is None:
    raise Exception("Missing `WEAVIATE_ENVIRONMENT` environment variable.")

if os.environ.get("WEAVIATE_URL", None) is None:
    raise Exception("Missing `WEAVIATE_URL` environment variable.")

if os.environ.get("OPENAI_API_KEY", None) is None:
    raise Exception("Missing `OPENAI_API_KEY` environment variable.")

# Initialize the retriever
WEAVIATE_INDEX_NAME = os.environ.get("WEAVIATE_INDEX", "langchain-test")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
auth_client_secret = (weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),)
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
    },
)
retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name=WEAVIATE_INDEX_NAME,
    text_key="text",
    attributes=[],
    create_schema_if_missing=True,
)

# # Ingest code - you may need to run this the first time
# # Load
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()
#
# # Split
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)
#
# # Add to vectorDB
# retriever.add_documents(all_splits)


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
