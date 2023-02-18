"""Wrappers on top of large language models APIs."""
from typing import Dict, Type

from langchain.llms.ai21 import AI21
from langchain.llms.anthropic import Anthropic
from langchain.llms.base import BaseLLM
from langchain.llms.cerebriumai import CerebriumAI
from langchain.llms.cohere import Cohere
from langchain.llms.forefrontai import ForefrontAI
from langchain.llms.gooseai import GooseAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.openai import AzureOpenAI, OpenAI
from langchain.llms.petals import Petals
from langchain.llms.promptlayer_openai import PromptLayerOpenAI

__all__ = [
    "Anthropic",
    "CerebriumAI",
    "Cohere",
    "ForefrontAI",
    "GooseAI",
    "NLPCloud",
    "OpenAI",
    "Petals",
    "HuggingFaceHub",
    "HuggingFacePipeline",
    "AI21",
    "AzureOpenAI",
    "PromptLayerOpenAI",
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "ai21": AI21,
    "anthropic": Anthropic,
    "cerebriumai": CerebriumAI,
    "cohere": Cohere,
    "forefrontai": ForefrontAI,
    "gooseai": GooseAI,
    "huggingface_hub": HuggingFaceHub,
    "nlpcloud": NLPCloud,
    "openai": OpenAI,
    "petals": Petals,
    "huggingface_pipeline": HuggingFacePipeline,
    "azure": AzureOpenAI,
}
