"""Wrappers around embedding modules."""
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings

__all__ = ["OpenAIEmbeddings", "HuggingFaceEmbeddings", "CohereEmbeddings"]
