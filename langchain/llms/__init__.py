"""Wrappers on top of large language models APIs."""
from typing import Dict, Type

from langchain.llms.ai21 import AI21
from langchain.llms.aleph_alpha import AlephAlpha
from langchain.llms.anthropic import Anthropic
from langchain.llms.base import BaseLLM
from langchain.llms.cerebriumai import CerebriumAI
from langchain.llms.cohere import Cohere
from langchain.llms.deepinfra import DeepInfra
from langchain.llms.forefrontai import ForefrontAI
from langchain.llms.gooseai import GooseAI
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.openai import AzureOpenAI, OpenAI
from langchain.llms.petals import Petals
from langchain.llms.promptlayer_openai import PromptLayerOpenAI
from langchain.llms.self_hosted import SelfHostedPipeline
from langchain.llms.self_hosted_hugging_face import SelfHostedHuggingFaceLLM

__all__ = [
    "Anthropic",
    "AlephAlpha",
    "CerebriumAI",
    "Cohere",
    "DeepInfra",
    "ForefrontAI",
    "GooseAI",
    "NLPCloud",
    "OpenAI",
    "Petals",
    "HuggingFaceEndpoint",
    "HuggingFaceHub",
    "HuggingFacePipeline",
    "AI21",
    "AzureOpenAI",
    "SelfHostedPipeline",
    "SelfHostedHuggingFaceLLM",
    "PromptLayerOpenAI",
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "ai21": AI21,
    "aleph_alpha": AlephAlpha,
    "anthropic": Anthropic,
    "cerebriumai": CerebriumAI,
    "cohere": Cohere,
    "deepinfra": DeepInfra,
    "forefrontai": ForefrontAI,
    "gooseai": GooseAI,
    "huggingface_hub": HuggingFaceHub,
    "huggingface_endpoint": HuggingFaceEndpoint,
    "nlpcloud": NLPCloud,
    "openai": OpenAI,
    "petals": Petals,
    "huggingface_pipeline": HuggingFacePipeline,
    "azure": AzureOpenAI,
    "self_hosted": SelfHostedPipeline,
    "self_hosted_hugging_face": SelfHostedHuggingFaceLLM,
}
