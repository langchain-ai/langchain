"""Wrappers on top of large language models APIs."""
from typing import Dict, Type

from langchain.llms.ai21 import AI21
from langchain.llms.aleph_alpha import AlephAlpha
from langchain.llms.anthropic import Anthropic
from langchain.llms.bananadev import Banana
from langchain.llms.base import BaseLLM
from langchain.llms.cerebriumai import CerebriumAI
from langchain.llms.cohere import Cohere
from langchain.llms.deepinfra import DeepInfra
from langchain.llms.forefrontai import ForefrontAI
from langchain.llms.gooseai import GooseAI
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.llamacpp import LlamaCpp
from langchain.llms.modal import Modal
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.openai import AzureOpenAI, OpenAI, OpenAIChat
from langchain.llms.petals import Petals
from langchain.llms.promptlayer_openai import PromptLayerOpenAI, PromptLayerOpenAIChat
from langchain.llms.replicate import Replicate
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.llms.self_hosted import SelfHostedPipeline
from langchain.llms.self_hosted_hugging_face import SelfHostedHuggingFaceLLM
from langchain.llms.stochasticai import StochasticAI
from langchain.llms.writer import Writer

__all__ = [
    "Anthropic",
    "AlephAlpha",
    "Banana",
    "CerebriumAI",
    "Cohere",
    "DeepInfra",
    "ForefrontAI",
    "GooseAI",
    "LlamaCpp",
    "Modal",
    "NLPCloud",
    "OpenAI",
    "OpenAIChat",
    "Petals",
    "HuggingFaceEndpoint",
    "HuggingFaceHub",
    "SagemakerEndpoint",
    "HuggingFacePipeline",
    "AI21",
    "AzureOpenAI",
    "Replicate",
    "SelfHostedPipeline",
    "SelfHostedHuggingFaceLLM",
    "PromptLayerOpenAI",
    "PromptLayerOpenAIChat",
    "StochasticAI",
    "Writer",
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "ai21": AI21,
    "aleph_alpha": AlephAlpha,
    "anthropic": Anthropic,
    "bananadev": Banana,
    "cerebriumai": CerebriumAI,
    "cohere": Cohere,
    "deepinfra": DeepInfra,
    "forefrontai": ForefrontAI,
    "gooseai": GooseAI,
    "huggingface_hub": HuggingFaceHub,
    "huggingface_endpoint": HuggingFaceEndpoint,
    "llamacpp": LlamaCpp,
    "modal": Modal,
    "sagemaker_endpoint": SagemakerEndpoint,
    "nlpcloud": NLPCloud,
    "openai": OpenAI,
    "petals": Petals,
    "huggingface_pipeline": HuggingFacePipeline,
    "azure": AzureOpenAI,
    "replicate": Replicate,
    "self_hosted": SelfHostedPipeline,
    "self_hosted_hugging_face": SelfHostedHuggingFaceLLM,
    "stochasticai": StochasticAI,
    "writer": Writer,
}
