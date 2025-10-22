"""Module for OpenAI embeddings."""

from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings

__all__ = ["AzureOpenAIEmbeddings", "OpenAIEmbeddings"]
