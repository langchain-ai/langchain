"""**Embedding models**.

**Embedding models**  are wrappers around embedding models
from different APIs and services.

Embedding models can be LLMs or not.
"""

import logging
from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer
from langchain_classic.embeddings.base import init_embeddings
from langchain_classic.embeddings.cache import CacheBackedEmbeddings

if TYPE_CHECKING:
    from langchain_community.embeddings import (
        AlephAlphaAsymmetricSemanticEmbedding,
        AlephAlphaSymmetricSemanticEmbedding,
        AwaEmbeddings,
        AzureOpenAIEmbeddings,
        BedrockEmbeddings,
        BookendEmbeddings,
        ClarifaiEmbeddings,
        CohereEmbeddings,
        DashScopeEmbeddings,
        DatabricksEmbeddings,
        DeepInfraEmbeddings,
        DeterministicFakeEmbedding,
        EdenAiEmbeddings,
        ElasticsearchEmbeddings,
        EmbaasEmbeddings,
        ErnieEmbeddings,
        FakeEmbeddings,
        FastEmbedEmbeddings,
        GooglePalmEmbeddings,
        GPT4AllEmbeddings,
        GradientEmbeddings,
        HuggingFaceBgeEmbeddings,
        HuggingFaceEmbeddings,
        HuggingFaceHubEmbeddings,
        HuggingFaceInferenceAPIEmbeddings,
        HuggingFaceInstructEmbeddings,
        InfinityEmbeddings,
        JavelinAIGatewayEmbeddings,
        JinaEmbeddings,
        JohnSnowLabsEmbeddings,
        LlamaCppEmbeddings,
        LocalAIEmbeddings,
        MiniMaxEmbeddings,
        MlflowAIGatewayEmbeddings,
        MlflowEmbeddings,
        ModelScopeEmbeddings,
        MosaicMLInstructorEmbeddings,
        NLPCloudEmbeddings,
        OctoAIEmbeddings,
        OllamaEmbeddings,
        OpenAIEmbeddings,
        OpenVINOEmbeddings,
        QianfanEmbeddingsEndpoint,
        SagemakerEndpointEmbeddings,
        SelfHostedEmbeddings,
        SelfHostedHuggingFaceEmbeddings,
        SelfHostedHuggingFaceInstructEmbeddings,
        SentenceTransformerEmbeddings,
        SpacyEmbeddings,
        TensorflowHubEmbeddings,
        VertexAIEmbeddings,
        VoyageEmbeddings,
        XinferenceEmbeddings,
    )


logger = logging.getLogger(__name__)


# TODO: this is in here to maintain backwards compatibility
class HypotheticalDocumentEmbedder:
    def __init__(self, *args: Any, **kwargs: Any):
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain_classic.chains import "
            "HypotheticalDocumentEmbedder` instead",
        )
        from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H(*args, **kwargs)  # type: ignore[return-value] # noqa: PLE0101

    @classmethod
    def from_llm(cls, *args: Any, **kwargs: Any) -> Any:
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain_classic.chains import "
            "HypotheticalDocumentEmbedder` instead",
        )
        from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H.from_llm(*args, **kwargs)


# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "AlephAlphaAsymmetricSemanticEmbedding": "langchain_community.embeddings",
    "AlephAlphaSymmetricSemanticEmbedding": "langchain_community.embeddings",
    "AwaEmbeddings": "langchain_community.embeddings",
    "AzureOpenAIEmbeddings": "langchain_community.embeddings",
    "BedrockEmbeddings": "langchain_community.embeddings",
    "BookendEmbeddings": "langchain_community.embeddings",
    "ClarifaiEmbeddings": "langchain_community.embeddings",
    "CohereEmbeddings": "langchain_community.embeddings",
    "DashScopeEmbeddings": "langchain_community.embeddings",
    "DatabricksEmbeddings": "langchain_community.embeddings",
    "DeepInfraEmbeddings": "langchain_community.embeddings",
    "DeterministicFakeEmbedding": "langchain_community.embeddings",
    "EdenAiEmbeddings": "langchain_community.embeddings",
    "ElasticsearchEmbeddings": "langchain_community.embeddings",
    "EmbaasEmbeddings": "langchain_community.embeddings",
    "ErnieEmbeddings": "langchain_community.embeddings",
    "FakeEmbeddings": "langchain_community.embeddings",
    "FastEmbedEmbeddings": "langchain_community.embeddings",
    "GooglePalmEmbeddings": "langchain_community.embeddings",
    "GPT4AllEmbeddings": "langchain_community.embeddings",
    "GradientEmbeddings": "langchain_community.embeddings",
    "HuggingFaceBgeEmbeddings": "langchain_community.embeddings",
    "HuggingFaceEmbeddings": "langchain_community.embeddings",
    "HuggingFaceHubEmbeddings": "langchain_community.embeddings",
    "HuggingFaceInferenceAPIEmbeddings": "langchain_community.embeddings",
    "HuggingFaceInstructEmbeddings": "langchain_community.embeddings",
    "InfinityEmbeddings": "langchain_community.embeddings",
    "JavelinAIGatewayEmbeddings": "langchain_community.embeddings",
    "JinaEmbeddings": "langchain_community.embeddings",
    "JohnSnowLabsEmbeddings": "langchain_community.embeddings",
    "LlamaCppEmbeddings": "langchain_community.embeddings",
    "LocalAIEmbeddings": "langchain_community.embeddings",
    "MiniMaxEmbeddings": "langchain_community.embeddings",
    "MlflowAIGatewayEmbeddings": "langchain_community.embeddings",
    "MlflowEmbeddings": "langchain_community.embeddings",
    "ModelScopeEmbeddings": "langchain_community.embeddings",
    "MosaicMLInstructorEmbeddings": "langchain_community.embeddings",
    "NLPCloudEmbeddings": "langchain_community.embeddings",
    "OctoAIEmbeddings": "langchain_community.embeddings",
    "OllamaEmbeddings": "langchain_community.embeddings",
    "OpenAIEmbeddings": "langchain_community.embeddings",
    "OpenVINOEmbeddings": "langchain_community.embeddings",
    "QianfanEmbeddingsEndpoint": "langchain_community.embeddings",
    "SagemakerEndpointEmbeddings": "langchain_community.embeddings",
    "SelfHostedEmbeddings": "langchain_community.embeddings",
    "SelfHostedHuggingFaceEmbeddings": "langchain_community.embeddings",
    "SelfHostedHuggingFaceInstructEmbeddings": "langchain_community.embeddings",
    "SentenceTransformerEmbeddings": "langchain_community.embeddings",
    "SpacyEmbeddings": "langchain_community.embeddings",
    "TensorflowHubEmbeddings": "langchain_community.embeddings",
    "VertexAIEmbeddings": "langchain_community.embeddings",
    "VoyageEmbeddings": "langchain_community.embeddings",
    "XinferenceEmbeddings": "langchain_community.embeddings",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AlephAlphaAsymmetricSemanticEmbedding",
    "AlephAlphaSymmetricSemanticEmbedding",
    "AwaEmbeddings",
    "AzureOpenAIEmbeddings",
    "BedrockEmbeddings",
    "BookendEmbeddings",
    "CacheBackedEmbeddings",
    "ClarifaiEmbeddings",
    "CohereEmbeddings",
    "DashScopeEmbeddings",
    "DatabricksEmbeddings",
    "DeepInfraEmbeddings",
    "DeterministicFakeEmbedding",
    "EdenAiEmbeddings",
    "ElasticsearchEmbeddings",
    "EmbaasEmbeddings",
    "ErnieEmbeddings",
    "FakeEmbeddings",
    "FastEmbedEmbeddings",
    "GPT4AllEmbeddings",
    "GooglePalmEmbeddings",
    "GradientEmbeddings",
    "HuggingFaceBgeEmbeddings",
    "HuggingFaceEmbeddings",
    "HuggingFaceHubEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
    "HuggingFaceInstructEmbeddings",
    "InfinityEmbeddings",
    "JavelinAIGatewayEmbeddings",
    "JinaEmbeddings",
    "JohnSnowLabsEmbeddings",
    "LlamaCppEmbeddings",
    "LocalAIEmbeddings",
    "MiniMaxEmbeddings",
    "MlflowAIGatewayEmbeddings",
    "MlflowEmbeddings",
    "ModelScopeEmbeddings",
    "MosaicMLInstructorEmbeddings",
    "NLPCloudEmbeddings",
    "OctoAIEmbeddings",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    "OpenVINOEmbeddings",
    "QianfanEmbeddingsEndpoint",
    "SagemakerEndpointEmbeddings",
    "SelfHostedEmbeddings",
    "SelfHostedHuggingFaceEmbeddings",
    "SelfHostedHuggingFaceInstructEmbeddings",
    "SentenceTransformerEmbeddings",
    "SpacyEmbeddings",
    "TensorflowHubEmbeddings",
    "VertexAIEmbeddings",
    "VoyageEmbeddings",
    "XinferenceEmbeddings",
    "init_embeddings",
]
