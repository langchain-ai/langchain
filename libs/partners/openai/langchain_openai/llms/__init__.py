"""Module for OpenAI large language models. Chat models are in `chat_models/`."""

from langchain_openai.llms.azure import AzureOpenAI
from langchain_openai.llms.base import OpenAI

__all__ = ["AzureOpenAI", "OpenAI"]
