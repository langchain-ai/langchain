"""
**LLM** classes provide
access to the large language model (**LLM**) APIs and services.

**Class hierarchy:**

.. code-block::

    BaseLanguageModel --> BaseLLM --> LLM --> <name>  # Examples: AI21, HuggingFaceHub, OpenAI

**Main helpers:**

.. code-block::

    LLMResult, PromptValue,
    CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun,
    CallbackManager, AsyncCallbackManager,
    AIMessage, BaseMessage
"""  # noqa: E501
from typing import Dict, Type

from langchain_xfyun.llms.ai21 import AI21
from langchain_xfyun.llms.aleph_alpha import AlephAlpha
from langchain_xfyun.llms.amazon_api_gateway import AmazonAPIGateway
from langchain_xfyun.llms.anthropic import Anthropic
from langchain_xfyun.llms.anyscale import Anyscale
from langchain_xfyun.llms.aviary import Aviary
from langchain_xfyun.llms.azureml_endpoint import AzureMLOnlineEndpoint
from langchain_xfyun.llms.bananadev import Banana
from langchain_xfyun.llms.base import BaseLLM
from langchain_xfyun.llms.baseten import Baseten
from langchain_xfyun.llms.beam import Beam
from langchain_xfyun.llms.bedrock import Bedrock
from langchain_xfyun.llms.bittensor import NIBittensorLLM
from langchain_xfyun.llms.cerebriumai import CerebriumAI
from langchain_xfyun.llms.chatglm import ChatGLM
from langchain_xfyun.llms.clarifai import Clarifai
from langchain_xfyun.llms.cohere import Cohere
from langchain_xfyun.llms.ctransformers import CTransformers
from langchain_xfyun.llms.databricks import Databricks
from langchain_xfyun.llms.deepinfra import DeepInfra
from langchain_xfyun.llms.deepsparse import DeepSparse
from langchain_xfyun.llms.edenai import EdenAI
from langchain_xfyun.llms.fake import FakeListLLM
from langchain_xfyun.llms.fireworks import Fireworks, FireworksChat
from langchain_xfyun.llms.forefrontai import ForefrontAI
from langchain_xfyun.llms.google_palm import GooglePalm
from langchain_xfyun.llms.gooseai import GooseAI
from langchain_xfyun.llms.gpt4all import GPT4All
from langchain_xfyun.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_xfyun.llms.huggingface_hub import HuggingFaceHub
from langchain_xfyun.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_xfyun.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_xfyun.llms.human import HumanInputLLM
from langchain_xfyun.llms.koboldai import KoboldApiLLM
from langchain_xfyun.llms.llamacpp import LlamaCpp
from langchain_xfyun.llms.manifest import ManifestWrapper
from langchain_xfyun.llms.minimax import Minimax
from langchain_xfyun.llms.mlflow_ai_gateway import MlflowAIGateway
from langchain_xfyun.llms.modal import Modal
from langchain_xfyun.llms.mosaicml import MosaicML
from langchain_xfyun.llms.nlpcloud import NLPCloud
from langchain_xfyun.llms.octoai_endpoint import OctoAIEndpoint
from langchain_xfyun.llms.ollama import Ollama
from langchain_xfyun.llms.openai import AzureOpenAI, OpenAI, OpenAIChat
from langchain_xfyun.llms.openllm import OpenLLM
from langchain_xfyun.llms.openlm import OpenLM
from langchain_xfyun.llms.petals import Petals
from langchain_xfyun.llms.pipelineai import PipelineAI
from langchain_xfyun.llms.predibase import Predibase
from langchain_xfyun.llms.predictionguard import PredictionGuard
from langchain_xfyun.llms.promptguard import PromptGuard
from langchain_xfyun.llms.promptlayer_openai import PromptLayerOpenAI, PromptLayerOpenAIChat
from langchain_xfyun.llms.replicate import Replicate
from langchain_xfyun.llms.rwkv import RWKV
from langchain_xfyun.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain_xfyun.llms.self_hosted import SelfHostedPipeline
from langchain_xfyun.llms.self_hosted_hugging_face import SelfHostedHuggingFaceLLM
from langchain_xfyun.llms.stochasticai import StochasticAI
from langchain_xfyun.llms.symblai_nebula import Nebula
from langchain_xfyun.llms.textgen import TextGen
from langchain_xfyun.llms.titan_takeoff import TitanTakeoff
from langchain_xfyun.llms.tongyi import Tongyi
from langchain_xfyun.llms.vertexai import VertexAI
from langchain_xfyun.llms.vllm import VLLM, VLLMOpenAI
from langchain_xfyun.llms.writer import Writer
from langchain_xfyun.llms.xinference import Xinference

__all__ = [
    "AI21",
    "AlephAlpha",
    "AmazonAPIGateway",
    "Anthropic",
    "Anyscale",
    "Aviary",
    "AzureMLOnlineEndpoint",
    "AzureOpenAI",
    "Banana",
    "Baseten",
    "Beam",
    "Bedrock",
    "CTransformers",
    "CerebriumAI",
    "ChatGLM",
    "Clarifai",
    "Cohere",
    "Databricks",
    "DeepInfra",
    "DeepSparse",
    "EdenAI",
    "FakeListLLM",
    "Fireworks",
    "FireworksChat",
    "ForefrontAI",
    "GPT4All",
    "GooglePalm",
    "GooseAI",
    "HuggingFaceEndpoint",
    "HuggingFaceHub",
    "HuggingFacePipeline",
    "HuggingFaceTextGenInference",
    "HumanInputLLM",
    "KoboldApiLLM",
    "LlamaCpp",
    "TextGen",
    "ManifestWrapper",
    "Minimax",
    "MlflowAIGateway",
    "Modal",
    "MosaicML",
    "Nebula",
    "NIBittensorLLM",
    "NLPCloud",
    "Ollama",
    "OpenAI",
    "OpenAIChat",
    "OpenLLM",
    "OpenLM",
    "Petals",
    "PipelineAI",
    "Predibase",
    "PredictionGuard",
    "PromptLayerOpenAI",
    "PromptLayerOpenAIChat",
    "PromptGuard",
    "RWKV",
    "Replicate",
    "SagemakerEndpoint",
    "SelfHostedHuggingFaceLLM",
    "SelfHostedPipeline",
    "StochasticAI",
    "TitanTakeoff",
    "Tongyi",
    "VertexAI",
    "VLLM",
    "VLLMOpenAI",
    "Writer",
    "OctoAIEndpoint",
    "Xinference",
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "ai21": AI21,
    "aleph_alpha": AlephAlpha,
    "amazon_api_gateway": AmazonAPIGateway,
    "amazon_bedrock": Bedrock,
    "anthropic": Anthropic,
    "anyscale": Anyscale,
    "aviary": Aviary,
    "azure": AzureOpenAI,
    "azureml_endpoint": AzureMLOnlineEndpoint,
    "bananadev": Banana,
    "baseten": Baseten,
    "beam": Beam,
    "cerebriumai": CerebriumAI,
    "chat_glm": ChatGLM,
    "clarifai": Clarifai,
    "cohere": Cohere,
    "ctransformers": CTransformers,
    "databricks": Databricks,
    "deepinfra": DeepInfra,
    "deepsparse": DeepSparse,
    "edenai": EdenAI,
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
    "koboldai": KoboldApiLLM,
    "llamacpp": LlamaCpp,
    "textgen": TextGen,
    "minimax": Minimax,
    "mlflow-ai-gateway": MlflowAIGateway,
    "modal": Modal,
    "mosaic": MosaicML,
    "nebula": Nebula,
    "nibittensor": NIBittensorLLM,
    "nlpcloud": NLPCloud,
    "ollama": Ollama,
    "openai": OpenAI,
    "openlm": OpenLM,
    "petals": Petals,
    "pipelineai": PipelineAI,
    "predibase": Predibase,
    "promptguard": PromptGuard,
    "replicate": Replicate,
    "rwkv": RWKV,
    "sagemaker_endpoint": SagemakerEndpoint,
    "self_hosted": SelfHostedPipeline,
    "self_hosted_hugging_face": SelfHostedHuggingFaceLLM,
    "stochasticai": StochasticAI,
    "tongyi": Tongyi,
    "titan_takeoff": TitanTakeoff,
    "vertexai": VertexAI,
    "openllm": OpenLLM,
    "openllm_client": OpenLLM,
    "vllm": VLLM,
    "vllm_openai": VLLMOpenAI,
    "writer": Writer,
    "xinference": Xinference,
}
