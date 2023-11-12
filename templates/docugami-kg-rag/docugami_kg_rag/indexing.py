## Ingest code - you may need to run this the first time
# Load
import os
from operator import itemgetter
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DocugamiLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain.vectorstores import Pinecone
from pydantic import BaseModel, Field
from langchain.document_loaders import WebBaseLoader

if __name__ == "__main__":
    DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")
    
    loader = DocugamiLoader(docset_id="4eik4b8kuoup", document_ids=["4vevuqrux0yb"])
    data = loader.load()

    # Add to vectorDB
    vectorstore = Pinecone.from_documents(
        documents=data, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME
    )
    retriever = vectorstore.as_retriever()

