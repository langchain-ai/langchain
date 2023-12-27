import os

from langchain.chat_models import ChatCohere
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

required_env_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "COHERE_API_KEY"]

for var in required_env_vars:
    if os.environ.get(var, None) is None:
        raise Exception(f"Missing `{var}` environment variable.")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")

# See https://docs.cohere.com/reference/embed for more models
EMBED_MODEL = "embed-english-v3.0"
cohere_embeddings = CohereEmbeddings(model=EMBED_MODEL)

# You may set this to False if you've already ingested the data
INGEST_DATA = True

if INGEST_DATA:
    # Load
    from langchain.document_loaders import WebBaseLoader

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()

    # Split
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # Add to vectorDB
    vectorstore = Pinecone.from_documents(
        documents=all_splits,
        embedding=cohere_embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    retriever = vectorstore.as_retriever()

vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, cohere_embeddings)

# Get k=10 docs
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Re-rank
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatCohere()
chain = (
    RunnableParallel(
        {"context": compression_retriever, "question": RunnablePassthrough()}
    )
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
