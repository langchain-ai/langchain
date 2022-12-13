"""Wrappers on top of large language models APIs."""
from typing import Dict, Type

from langchain.llms.ai21 import AI21
from langchain.llms.base import LLM
from langchain.llms.cohere import Cohere
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.openai import OpenAI

__all__ = ["Cohere", "NLPCloud", "OpenAI", "HuggingFaceHub", "AI21"]

type_to_cls_dict: Dict[str, Type[LLM]] = {
    "ai21": AI21,
    "cohere": Cohere,
    "huggingface_hub": HuggingFaceHub,
    "nlpcloud": NLPCloud,
    "openai": OpenAI,
}
