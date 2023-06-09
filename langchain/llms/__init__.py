"""Wrappers on top of large language models APIs."""
from typing import Dict, Type

from langchain.llms.ai21 import AI21
from langchain.llms.aleph_alpha import AlephAlpha
from langchain.llms.anthropic import Anthropic
from langchain.llms.anyscale import Anyscale
from langchain.llms.aviary import Aviary
from langchain.llms.bananadev import Banana
from langchain.llms.base import BaseLLM
from langchain.llms.baseten import Baseten
from langchain.llms.beam import Beam
from langchain.llms.bedrock import Bedrock
from langchain.llms.cerebriumai import CerebriumAI
from langchain.llms.cohere import Cohere
from langchain.llms.ctransformers import CTransformers
from langchain.llms.databricks import Databricks
from langchain.llms.deepinfra import DeepInfra
from langchain.llms.fake import FakeListLLM
from langchain.llms.forefrontai import ForefrontAI
from langchain.llms.google_palm import GooglePalm
from langchain.llms.gooseai import GooseAI
from langchain.llms.gpt4all import GPT4All
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain.llms.human import HumanInputLLM
from langchain.llms.llamacpp import LlamaCpp
from langchain.llms.modal import Modal
from langchain.llms.mosaicml import MosaicML
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.openai import AzureOpenAI, OpenAI, OpenAIChat
from langchain.llms.openlm import OpenLM
from langchain.llms.petals import Petals
from langchain.llms.pipelineai import PipelineAI
from langchain.llms.predictionguard import PredictionGuard
from langchain.llms.promptlayer_openai import PromptLayerOpenAI, PromptLayerOpenAIChat
from langchain.llms.replicate import Replicate
from langchain.llms.rwkv import RWKV
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.llms.self_hosted import SelfHostedPipeline
from langchain.llms.self_hosted_hugging_face import SelfHostedHuggingFaceLLM
from langchain.llms.stochasticai import StochasticAI
from langchain.llms.vertexai import VertexAI
from langchain.llms.writer import Writer

__all__ = [
    "Anthropic",
    "AlephAlpha",
    "Anyscale",
    "Aviary",
    "Banana",
    "Baseten",
    "Beam",
    "Bedrock",
    "CerebriumAI",
    "Cohere",
    "CTransformers",
    "Databricks",
    "DeepInfra",
    "ForefrontAI",
    "GooglePalm",
    "GooseAI",
    "GPT4All",
    "LlamaCpp",
    "Modal",
    "MosaicML",
    "NLPCloud",
    "OpenAI",
    "OpenAIChat",
    "OpenLM",
    "Petals",
    "PipelineAI",
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
    "RWKV",
    "PredictionGuard",
    "HumanInputLLM",
    "HuggingFaceTextGenInference",
    "FakeListLLM",
    "VertexAI",
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "ai21": AI21,
    "aleph_alpha": AlephAlpha,
    "anthropic": Anthropic,
    "anyscale": Anyscale,
    "aviary": Aviary,
    "bananadev": Banana,
    "baseten": Baseten,
    "beam": Beam,
    "cerebriumai": CerebriumAI,
    "cohere": Cohere,
    "ctransformers": CTransformers,
    "databricks": Databricks,
    "deepinfra": DeepInfra,
    "forefrontai": ForefrontAI,
    "google_palm": GooglePalm,
    "gooseai": GooseAI,
    "gpt4all": GPT4All,
    "huggingface_hub": HuggingFaceHub,
    "huggingface_endpoint": HuggingFaceEndpoint,
    "llamacpp": LlamaCpp,
    "modal": Modal,
    "mosaic": MosaicML,
    "sagemaker_endpoint": SagemakerEndpoint,
    "nlpcloud": NLPCloud,
    "human-input": HumanInputLLM,
    "openai": OpenAI,
    "openlm": OpenLM,
    "petals": Petals,
    "pipelineai": PipelineAI,
    "huggingface_pipeline": HuggingFacePipeline,
    "azure": AzureOpenAI,
    "replicate": Replicate,
    "self_hosted": SelfHostedPipeline,
    "self_hosted_hugging_face": SelfHostedHuggingFaceLLM,
    "stochasticai": StochasticAI,
    "writer": Writer,
    "rwkv": RWKV,
    "huggingface_textgen_inference": HuggingFaceTextGenInference,
    "fake-list": FakeListLLM,
    "vertexai": VertexAI,
}
