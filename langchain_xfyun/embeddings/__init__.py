"""**Embedding models**  are wrappers around embedding models
from different APIs and services.

**Embedding models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    Embeddings --> <name>Embeddings  # Examples: OpenAIEmbeddings, HuggingFaceEmbeddings
"""


import logging
from typing import Any

from langchain_xfyun.embeddings.aleph_alpha import (
    AlephAlphaAsymmetricSemanticEmbedding,
    AlephAlphaSymmetricSemanticEmbedding,
)
from langchain_xfyun.embeddings.awa import AwaEmbeddings
from langchain_xfyun.embeddings.bedrock import BedrockEmbeddings
from langchain_xfyun.embeddings.cache import CacheBackedEmbeddings
from langchain_xfyun.embeddings.clarifai import ClarifaiEmbeddings
from langchain_xfyun.embeddings.cohere import CohereEmbeddings
from langchain_xfyun.embeddings.dashscope import DashScopeEmbeddings
from langchain_xfyun.embeddings.deepinfra import DeepInfraEmbeddings
from langchain_xfyun.embeddings.edenai import EdenAiEmbeddings
from langchain_xfyun.embeddings.elasticsearch import ElasticsearchEmbeddings
from langchain_xfyun.embeddings.embaas import EmbaasEmbeddings
from langchain_xfyun.embeddings.ernie import ErnieEmbeddings
from langchain_xfyun.embeddings.fake import DeterministicFakeEmbedding, FakeEmbeddings
from langchain_xfyun.embeddings.google_palm import GooglePalmEmbeddings
from langchain_xfyun.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_xfyun.embeddings.huggingface import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain_xfyun.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_xfyun.embeddings.jina import JinaEmbeddings
from langchain_xfyun.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_xfyun.embeddings.localai import LocalAIEmbeddings
from langchain_xfyun.embeddings.minimax import MiniMaxEmbeddings
from langchain_xfyun.embeddings.mlflow_gateway import MlflowAIGatewayEmbeddings
from langchain_xfyun.embeddings.modelscope_hub import ModelScopeEmbeddings
from langchain_xfyun.embeddings.mosaicml import MosaicMLInstructorEmbeddings
from langchain_xfyun.embeddings.nlpcloud import NLPCloudEmbeddings
from langchain_xfyun.embeddings.octoai_embeddings import OctoAIEmbeddings
from langchain_xfyun.embeddings.openai import OpenAIEmbeddings
from langchain_xfyun.embeddings.sagemaker_endpoint import SagemakerEndpointEmbeddings
from langchain_xfyun.embeddings.self_hosted import SelfHostedEmbeddings
from langchain_xfyun.embeddings.self_hosted_hugging_face import (
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)
from langchain_xfyun.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_xfyun.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_xfyun.embeddings.tensorflow_hub import TensorflowHubEmbeddings
from langchain_xfyun.embeddings.vertexai import VertexAIEmbeddings
from langchain_xfyun.embeddings.xinference import XinferenceEmbeddings

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAIEmbeddings",
    "CacheBackedEmbeddings",
    "ClarifaiEmbeddings",
    "CohereEmbeddings",
    "ElasticsearchEmbeddings",
    "HuggingFaceEmbeddings",
    "JinaEmbeddings",
    "LlamaCppEmbeddings",
    "HuggingFaceHubEmbeddings",
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
]


# TODO: this is in here to maintain backwards compatibility
class HypotheticalDocumentEmbedder:
    def __init__(self, *args: Any, **kwargs: Any):
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain_xfyun.chains import HypotheticalDocumentEmbedder` instead"
        )
        from langchain_xfyun.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H(*args, **kwargs)  # type: ignore

    @classmethod
    def from_llm(cls, *args: Any, **kwargs: Any) -> Any:
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain_xfyun.chains import HypotheticalDocumentEmbedder` instead"
        )
        from langchain_xfyun.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H.from_llm(*args, **kwargs)
