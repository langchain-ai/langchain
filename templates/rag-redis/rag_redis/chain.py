import os

from langchain.vectorstores import Redis
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.pydantic_v1 import BaseModel

from rag_redis.config import (
    REDIS_URL,
    INDEX_NAME,
    INDEX_SCHEMA,
    DEBUG,
    EMBED_MODEL,
)

# Make this look better in the docs.
class Question(BaseModel):
    __root__: str


# Init Embeddings
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Connect to pre-loaded vectorstore -- run the ingest.py script to populate this
vectorstore = Redis.from_existing_index(
    embedding=embedder,
    index_name=INDEX_NAME,
    schema=INDEX_SCHEMA,
    redis_url=REDIS_URL
)
# TODO set parameters
retriever = vectorstore.as_retriever()


# Define our prompt
template = """
Use the following pieces of context from financial 10k filings data to answer the user question.
If you don't know the answer, say that you don't know, don't try to make up an answer.

This should be in the following format:

Question: [question here]
Answer: [answer here]

Begin!

Context:
---------
{context}
---------
Question: {question}
Answer:"""

prompt = PromptTemplate.from_template(template)

# RAG Chain
model = OpenAI()
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
).with_types(input_type=Question)