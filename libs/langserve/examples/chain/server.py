#!/usr/bin/env python
"""Example LangChain server exposes a chain composed of a prompt and an LLM."""
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

from langserve import add_routes

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


class ChainInput(TypedDict):
    """The input to the chain."""

    topic: str
    """The topic of the joke."""


add_routes(app, chain, input_type=ChainInput)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
