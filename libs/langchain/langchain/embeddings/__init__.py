"""**Embedding models**  are wrappers around embedding models
from different APIs and services.

**Embedding models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    Embeddings --> <name>Embeddings  # Examples: OpenAIEmbeddings, HuggingFaceEmbeddings
"""


import logging
from typing import Any

from langchain.embeddings.aleph_alpha import (
    AlephAlphaAsymmetricSemanticEmbedding,
    AlephAlphaSymmetricSemanticEmbedding,
)
from langchain.embeddings.awa import AwaEmbeddings
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.embeddings.clarifai import ClarifaiEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.embeddings.deepinfra import DeepInfraEmbeddings
from langchain.embeddings.edenai import EdenAiEmbeddings
from langchain.embeddings.elasticsearch import ElasticsearchEmbeddings
from langchain.embeddings.embaas import EmbaasEmbeddings
from langchain.embeddings.ernie import ErnieEmbeddings
from langchain.embeddings.fake import DeterministicFakeEmbedding, FakeEmbeddings
from langchain.embeddings.google_palm import GooglePalmEmbeddings
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.embeddings.huggingface import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain.embeddings.jina import JinaEmbeddings
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.embeddings.localai import LocalAIEmbeddings
from langchain.embeddings.minimax import MiniMaxEmbeddings
from langchain.embeddings.mlflow_gateway import MlflowAIGatewayEmbeddings
from langchain.embeddings.modelscope_hub import ModelScopeEmbeddings
from langchain.embeddings.mosaicml import MosaicMLInstructorEmbeddings
from langchain.embeddings.nlpcloud import NLPCloudEmbeddings
from langchain.embeddings.octoai_embeddings import OctoAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sagemaker_endpoint import SagemakerEndpointEmbeddings
from langchain.embeddings.self_hosted import SelfHostedEmbeddings
from langchain.embeddings.self_hosted_hugging_face import (
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.embeddings.tensorflow_hub import TensorflowHubEmbeddings
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.embeddings.xinference import XinferenceEmbeddings

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAIEmbeddings",
    "CacheBackedEmbeddings",
    "ClarifaiEmbeddings",
    "CohereEmbeddings",
    "ElasticsearchEmbeddings",
    "HuggingFaceEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
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
