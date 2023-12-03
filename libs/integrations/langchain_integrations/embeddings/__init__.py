"""**Embedding models**  are wrappers around embedding models
from different APIs and services.

**Embedding models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    Embeddings --> <name>Embeddings  # Examples: OpenAIEmbeddings, HuggingFaceEmbeddings
"""


import logging
from typing import Any

from langchain_integrations.embeddings.aleph_alpha import (
    AlephAlphaAsymmetricSemanticEmbedding,
    AlephAlphaSymmetricSemanticEmbedding,
)
from langchain_integrations.embeddings.awa import AwaEmbeddings
from langchain_integrations.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain_integrations.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from langchain_integrations.embeddings.bedrock import BedrockEmbeddings
from langchain_integrations.embeddings.cache import CacheBackedEmbeddings
from langchain_integrations.embeddings.clarifai import ClarifaiEmbeddings
from langchain_integrations.embeddings.cohere import CohereEmbeddings
from langchain_integrations.embeddings.dashscope import DashScopeEmbeddings
from langchain_integrations.embeddings.databricks import DatabricksEmbeddings
from langchain_integrations.embeddings.deepinfra import DeepInfraEmbeddings
from langchain_integrations.embeddings.edenai import EdenAiEmbeddings
from langchain_integrations.embeddings.elasticsearch import ElasticsearchEmbeddings
from langchain_integrations.embeddings.embaas import EmbaasEmbeddings
from langchain_integrations.embeddings.ernie import ErnieEmbeddings
from langchain_integrations.embeddings.fake import DeterministicFakeEmbedding, FakeEmbeddings
from langchain_integrations.embeddings.fastembed import FastEmbedEmbeddings
from langchain_integrations.embeddings.google_palm import GooglePalmEmbeddings
from langchain_integrations.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_integrations.embeddings.gradient_ai import GradientEmbeddings
from langchain_integrations.embeddings.huggingface import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain_integrations.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_integrations.embeddings.infinity import InfinityEmbeddings
from langchain_integrations.embeddings.javelin_ai_gateway import JavelinAIGatewayEmbeddings
from langchain_integrations.embeddings.jina import JinaEmbeddings
from langchain_integrations.embeddings.johnsnowlabs import JohnSnowLabsEmbeddings
from langchain_integrations.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_integrations.embeddings.localai import LocalAIEmbeddings
from langchain_integrations.embeddings.minimax import MiniMaxEmbeddings
from langchain_integrations.embeddings.mlflow import MlflowEmbeddings
from langchain_integrations.embeddings.mlflow_gateway import MlflowAIGatewayEmbeddings
from langchain_integrations.embeddings.modelscope_hub import ModelScopeEmbeddings
from langchain_integrations.embeddings.mosaicml import MosaicMLInstructorEmbeddings
from langchain_integrations.embeddings.nlpcloud import NLPCloudEmbeddings
from langchain_integrations.embeddings.octoai_embeddings import OctoAIEmbeddings
from langchain_integrations.embeddings.ollama import OllamaEmbeddings
from langchain_openai.embedding import OpenAIEmbeddings
from langchain_integrations.embeddings.sagemaker_endpoint import SagemakerEndpointEmbeddings
from langchain_integrations.embeddings.self_hosted import SelfHostedEmbeddings
from langchain_integrations.embeddings.self_hosted_hugging_face import (
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)
from langchain_integrations.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_integrations.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_integrations.embeddings.tensorflow_hub import TensorflowHubEmbeddings
from langchain_integrations.embeddings.vertexai import VertexAIEmbeddings
from langchain_integrations.embeddings.voyageai import VoyageEmbeddings
from langchain_integrations.embeddings.xinference import XinferenceEmbeddings

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAIEmbeddings",
    "AzureOpenAIEmbeddings",
    "CacheBackedEmbeddings",
    "ClarifaiEmbeddings",
    "CohereEmbeddings",
    "DatabricksEmbeddings",
    "ElasticsearchEmbeddings",
    "FastEmbedEmbeddings",
    "HuggingFaceEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
    "InfinityEmbeddings",
    "GradientEmbeddings",
    "JinaEmbeddings",
    "LlamaCppEmbeddings",
    "HuggingFaceHubEmbeddings",
    "MlflowEmbeddings",
    "MlflowAIGatewayEmbeddings",
    "ModelScopeEmbeddings",
    "TensorflowHubEmbeddings",
    "SagemakerEndpointEmbeddings",
    "HuggingFaceInstructEmbeddings",
    "MosaicMLInstructorEmbeddings",
    "SelfHostedEmbeddings",
    "SelfHostedHuggingFaceEmbeddings",
    "SelfHostedHuggingFaceInstructEmbeddings",
    "FakeEmbeddings",
    "DeterministicFakeEmbedding",
    "AlephAlphaAsymmetricSemanticEmbedding",
    "AlephAlphaSymmetricSemanticEmbedding",
    "SentenceTransformerEmbeddings",
    "GooglePalmEmbeddings",
    "MiniMaxEmbeddings",
    "VertexAIEmbeddings",
    "BedrockEmbeddings",
    "DeepInfraEmbeddings",
    "EdenAiEmbeddings",
    "DashScopeEmbeddings",
    "EmbaasEmbeddings",
    "OctoAIEmbeddings",
    "SpacyEmbeddings",
    "NLPCloudEmbeddings",
    "GPT4AllEmbeddings",
    "XinferenceEmbeddings",
    "LocalAIEmbeddings",
    "AwaEmbeddings",
    "HuggingFaceBgeEmbeddings",
    "ErnieEmbeddings",
    "JavelinAIGatewayEmbeddings",
    "OllamaEmbeddings",
    "QianfanEmbeddingsEndpoint",
    "JohnSnowLabsEmbeddings",
    "VoyageEmbeddings",
]


# TODO: this is in here to maintain backwards compatibility
class HypotheticalDocumentEmbedder:
    def __init__(self, *args: Any, **kwargs: Any):
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain_integrations.chains import HypotheticalDocumentEmbedder` instead"
        )
        from langchain_integrations.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H(*args, **kwargs)  # type: ignore

    @classmethod
    def from_llm(cls, *args: Any, **kwargs: Any) -> Any:
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain_integrations.chains import HypotheticalDocumentEmbedder` instead"
        )
        from langchain_integrations.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H.from_llm(*args, **kwargs)
