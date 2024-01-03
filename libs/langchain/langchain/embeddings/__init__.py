"""**Embedding models**  are wrappers around embedding models
from different APIs and services.

**Embedding models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    Embeddings --> <name>Embeddings  # Examples: OpenAIEmbeddings, HuggingFaceEmbeddings
"""


import logging
from typing import Any

from langchain_community.embeddings.aleph_alpha import (
    AlephAlphaAsymmetricSemanticEmbedding,
    AlephAlphaSymmetricSemanticEmbedding,
)
from langchain_community.embeddings.awa import AwaEmbeddings
from langchain_community.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings.baidu_qianfan_endpoint import (
    QianfanEmbeddingsEndpoint,
)
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.bookend import BookendEmbeddings
from langchain_community.embeddings.clarifai import ClarifaiEmbeddings
from langchain_community.embeddings.cohere import CohereEmbeddings
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.embeddings.databricks import DatabricksEmbeddings
from langchain_community.embeddings.deepinfra import DeepInfraEmbeddings
from langchain_community.embeddings.edenai import EdenAiEmbeddings
from langchain_community.embeddings.elasticsearch import ElasticsearchEmbeddings
from langchain_community.embeddings.embaas import EmbaasEmbeddings
from langchain_community.embeddings.ernie import ErnieEmbeddings
from langchain_community.embeddings.fake import (
    DeterministicFakeEmbedding,
    FakeEmbeddings,
)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.embeddings.gradient_ai import GradientEmbeddings
from langchain_community.embeddings.huggingface import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.embeddings.infinity import InfinityEmbeddings
from langchain_community.embeddings.javelin_ai_gateway import JavelinAIGatewayEmbeddings
from langchain_community.embeddings.jina import JinaEmbeddings
from langchain_community.embeddings.johnsnowlabs import JohnSnowLabsEmbeddings
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_community.embeddings.localai import LocalAIEmbeddings
from langchain_community.embeddings.minimax import MiniMaxEmbeddings
from langchain_community.embeddings.mlflow import MlflowEmbeddings
from langchain_community.embeddings.mlflow_gateway import MlflowAIGatewayEmbeddings
from langchain_community.embeddings.modelscope_hub import ModelScopeEmbeddings
from langchain_community.embeddings.mosaicml import MosaicMLInstructorEmbeddings
from langchain_community.embeddings.nlpcloud import NLPCloudEmbeddings
from langchain_community.embeddings.octoai_embeddings import OctoAIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import (
    SagemakerEndpointEmbeddings,
)
from langchain_community.embeddings.self_hosted import SelfHostedEmbeddings
from langchain_community.embeddings.self_hosted_hugging_face import (
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.embeddings.tensorflow_hub import TensorflowHubEmbeddings
from langchain_community.embeddings.vertexai import VertexAIEmbeddings
from langchain_community.embeddings.voyageai import VoyageEmbeddings
from langchain_community.embeddings.xinference import XinferenceEmbeddings

from langchain.embeddings.cache import CacheBackedEmbeddings

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
    "BookendEmbeddings",
]


# TODO: this is in here to maintain backwards compatibility
class HypotheticalDocumentEmbedder:
    def __init__(self, *args: Any, **kwargs: Any):
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain.chains import HypotheticalDocumentEmbedder` instead"
        )
        from langchain.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H(*args, **kwargs)  # type: ignore

    @classmethod
    def from_llm(cls, *args: Any, **kwargs: Any) -> Any:
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain.chains import HypotheticalDocumentEmbedder` instead"
        )
        from langchain.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H.from_llm(*args, **kwargs)
