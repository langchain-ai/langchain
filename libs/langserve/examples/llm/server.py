#!/usr/bin/env python
"""Example LangChain server exposes multiple runnables (LLMs in this case)."""
from typing import List, Union

from fastapi import FastAPI
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts.chat import ChatPromptValue
from langchain.schema.messages import HumanMessage, SystemMessage

from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

LLMInput = Union[List[Union[SystemMessage, HumanMessage, str]], str, ChatPromptValue]

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
    input_type=LLMInput,
    config_keys=[],
)
add_routes(
    app,
    ChatAnthropic(),
    path="/anthropic",
    input_type=LLMInput,
    config_keys=[],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
