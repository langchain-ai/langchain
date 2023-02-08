"""Wrappers on top of large language models APIs."""
from typing import Dict, Type

from langchain.llms.ai21 import AI21
from langchain.llms.anthropic import Anthropic
from langchain.llms.base import BaseLLM
from langchain.llms.cohere import Cohere
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.openai import AzureOpenAI, OpenAI

__all__ = [
    "Anthropic",
    "Cohere",
    "NLPCloud",
    "OpenAI",
    "HuggingFaceHub",
    "HuggingFacePipeline",
    "AI21",
    "AzureOpenAI",
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "ai21": AI21,
    "anthropic": Anthropic,
    "cohere": Cohere,
    "huggingface_hub": HuggingFaceHub,
    "nlpcloud": NLPCloud,
    "openai": OpenAI,
    "huggingface_pipeline": HuggingFacePipeline,
    "azure": AzureOpenAI,
}
