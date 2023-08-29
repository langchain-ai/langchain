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

from langchain.llms.ai21 import AI21
from langchain.llms.aleph_alpha import AlephAlpha
from langchain.llms.amazon_api_gateway import AmazonAPIGateway
from langchain.llms.anthropic import Anthropic
from langchain.llms.anyscale import Anyscale
from langchain.llms.aviary import Aviary
from langchain.llms.azureml_endpoint import AzureMLOnlineEndpoint
from langchain.llms.bananadev import Banana
from langchain.llms.base import BaseLLM
from langchain.llms.baseten import Baseten
from langchain.llms.beam import Beam
from langchain.llms.bedrock import Bedrock
from langchain.llms.bittensor import NIBittensorLLM
from langchain.llms.cerebriumai import CerebriumAI
from langchain.llms.chatglm import ChatGLM
from langchain.llms.clarifai import Clarifai
from langchain.llms.cohere import Cohere
from langchain.llms.ctransformers import CTransformers
from langchain.llms.databricks import Databricks
from langchain.llms.deepinfra import DeepInfra
from langchain.llms.deepsparse import DeepSparse
from langchain.llms.edenai import EdenAI
from langchain.llms.fake import FakeListLLM
from langchain.llms.fireworks import Fireworks, FireworksChat
from langchain.llms.forefrontai import ForefrontAI
from langchain.llms.google_palm import GooglePalm
from langchain.llms.gooseai import GooseAI
from langchain.llms.gpt4all import GPT4All
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain.llms.human import HumanInputLLM
from langchain.llms.koboldai import KoboldApiLLM
from langchain.llms.llamacpp import LlamaCpp
from langchain.llms.manifest import ManifestWrapper
from langchain.llms.minimax import Minimax
from langchain.llms.mlflow_ai_gateway import MlflowAIGateway
from langchain.llms.modal import Modal
from langchain.llms.mosaicml import MosaicML
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.octoai_endpoint import OctoAIEndpoint
from langchain.llms.ollama import Ollama
from langchain.llms.openai import AzureOpenAI, OpenAI, OpenAIChat
from langchain.llms.openllm import OpenLLM
from langchain.llms.openlm import OpenLM
from langchain.llms.petals import Petals
from langchain.llms.pipelineai import PipelineAI
from langchain.llms.predibase import Predibase
from langchain.llms.predictionguard import PredictionGuard
from langchain.llms.promptguard import PromptGuard
from langchain.llms.promptlayer_openai import PromptLayerOpenAI, PromptLayerOpenAIChat
from langchain.llms.replicate import Replicate
from langchain.llms.rwkv import RWKV
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.llms.self_hosted import SelfHostedPipeline
from langchain.llms.self_hosted_hugging_face import SelfHostedHuggingFaceLLM
from langchain.llms.stochasticai import StochasticAI
from langchain.llms.symblai_nebula import Nebula
from langchain.llms.textgen import TextGen
from langchain.llms.titan_takeoff import TitanTakeoff
from langchain.llms.tongyi import Tongyi
from langchain.llms.vertexai import VertexAI
from langchain.llms.vllm import VLLM, VLLMOpenAI
from langchain.llms.writer import Writer
from langchain.llms.xinference import Xinference

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
