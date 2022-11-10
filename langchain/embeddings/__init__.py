"""Wrappers around embedding modules."""
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

__all__ = ["OpenAIEmbeddings", "HuggingFaceEmbeddings", "CohereEmbeddings"]
