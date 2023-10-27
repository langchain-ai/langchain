import os

from typing import List

from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain.docstore.document import Document
from langchain.vectorstores.redis import RedisVectorStoreRetriever
from langchain.vectorstores import Redis
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel


from .redis import (
    REDIS_URL,
    INDEX_NAME,
    INDEX_SCHEMA,
    DEBUG,
)



# Check for openai API key
if "OPENAI_API_KEY" not in os.environ:
    raise Exception("Must provide an OPENAI_API_KEY as an env var.")


# Init Embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class Retriever(RedisVectorStoreRetriever):
    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        return self._get_relevant_documents(query=query, run_manager=run_manager)


# Set DEBUG env var to "true" if you wish to enable LC debugging module
if DEBUG:
    import langchain
    langchain.debug=True


# Check for openai API key
if "OPENAI_API_KEY" not in os.environ:
    raise Exception("Must provide an OPENAI_API_KEY as an env var.")


# Connect to pre-loaded vectorstore -- run the ingest.py script to populate this
vectorstore = Redis.from_existing_index(
    embedding=embedder,
    index_name=INDEX_NAME,
    schema=INDEX_SCHEMA,
    redis_url=REDIS_URL
)
retriever = Retriever(vectorstore=vectorstore)#vectorstore.as_retriever()


# Define our prompt
template = """Use the following pieces of context from financial 10k filings data to answer the user question at the end. If you don't know the answer, say that you don't know, don't try to make up an answer.

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
)