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
from langchain.llms.manifest import ManifestWrapper
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
from langchain.llms.textgen import TextGen
from langchain.llms.vertexai import VertexAI
from langchain.llms.writer import Writer

__all__ = [
    "AI21",
    "AlephAlpha",
    "Anthropic",
    "Anyscale",
    "Aviary",
    "AzureOpenAI",
    "Banana",
    "Baseten",
    "Beam",
    "Bedrock",
    "CTransformers",
    "CerebriumAI",
    "Cohere",
    "Databricks",
    "DeepInfra",
    "FakeListLLM",
    "ForefrontAI",
    "GPT4All",
    "GooglePalm",
    "GooseAI",
    "HuggingFaceEndpoint",
    "HuggingFaceHub",
    "HuggingFacePipeline",
    "HuggingFaceTextGenInference",
    "HumanInputLLM",
    "LlamaCpp",
    "TextGen",
    "ManifestWrapper",
    "Modal",
    "MosaicML",
    "NLPCloud",
    "OpenAI",
    "OpenAIChat",
    "OpenLM",
    "Petals",
    "PipelineAI",
    "PredictionGuard",
    "PromptLayerOpenAI",
    "PromptLayerOpenAIChat",
    "RWKV",
    "Replicate",
    "SagemakerEndpoint",
    "SelfHostedHuggingFaceLLM",
    "SelfHostedPipeline",
    "StochasticAI",
    "VertexAI",
    "Writer",
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "ai21": AI21,
    "aleph_alpha": AlephAlpha,
    "anthropic": Anthropic,
    "anyscale": Anyscale,
    "aviary": Aviary,
    "azure": AzureOpenAI,
    "bananadev": Banana,
    "baseten": Baseten,
    "beam": Beam,
    "cerebriumai": CerebriumAI,
    "cohere": Cohere,
    "ctransformers": CTransformers,
    "databricks": Databricks,
    "deepinfra": DeepInfra,
    "fake-list": FakeListLLM,
    "forefrontai": ForefrontAI,
    "google_palm": GooglePalm,
    "gooseai": GooseAI,
    "gpt4all": GPT4All,
    "huggingface_endpoint": HuggingFaceEndpoint,
    "huggingface_hub": HuggingFaceHub,
    "huggingface_pipeline": HuggingFacePipeline,
    "huggingface_textgen_inference": HuggingFaceTextGenInference,
    "human-input": HumanInputLLM,
    "llamacpp": LlamaCpp,
    "textgen": TextGen,
    "modal": Modal,
    "mosaic": MosaicML,
    "nlpcloud": NLPCloud,
    "openai": OpenAI,
    "openlm": OpenLM,
    "petals": Petals,
    "pipelineai": PipelineAI,
    "replicate": Replicate,
    "rwkv": RWKV,
    "sagemaker_endpoint": SagemakerEndpoint,
    "self_hosted": SelfHostedPipeline,
    "self_hosted_hugging_face": SelfHostedHuggingFaceLLM,
    "stochasticai": StochasticAI,
    "vertexai": VertexAI,
    "writer": Writer,
}
