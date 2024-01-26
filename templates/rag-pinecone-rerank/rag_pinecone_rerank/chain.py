import os

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.chat_models import ChatCohere, ChatOpenAI
from langchain_community.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
    raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")

EMBEDDING_TYPE = os.environ.get("EMBEDDING_TYPE", "OpenAI")

if EMBEDDING_TYPE == "OpenAI":
    print("Using OpenAI embeddings")
    if os.environ.get("OPENAI_API_KEY", None) is None:
        raise Exception("Missing `OPENAI_API_KEY` environment variable.")
    embeddings = OpenAIEmbeddings()
    chat_model = ChatOpenAI
elif EMBEDDING_TYPE == "Cohere":
    print("Using Cohere embeddings")
    if os.environ.get("COHERE_API_KEY", None) is None:
        raise Exception("Missing `COHERE_API_KEY` environment variable.")
    cohere_embed_model = "embed-english-v3.0"
    embeddings = CohereEmbeddings(model=cohere_embed_model)
    chat_model = ChatCohere
else:
    raise Exception("Invalid `EMBEDDING_TYPE` environment variable.")

### Ingest code - you may need to run this the first time
# # Load
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# # Split
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# # Add to vectorDB
# vectorstore = Pinecone.from_documents(
#     documents=all_splits, embedding=embeddings, index_name=PINECONE_INDEX_NAME
# )
# retriever = vectorstore.as_retriever()

vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)

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
model = chat_model()
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
