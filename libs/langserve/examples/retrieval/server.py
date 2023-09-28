#!/usr/bin/env python
"""Example LangChain server exposes a retrieval."""
from fastapi import FastAPI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langserve import add_routes

vectorstore = FAISS.from_texts(
    ["cats like fish", "dogs like sticks"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, retriever, input_type=str)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
