"""**Embedding models**  are wrappers around embedding models
from different APIs and services.

**Embedding models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    Embeddings --> <name>Embeddings  # Examples: OpenAIEmbeddings, HuggingFaceEmbeddings
"""


import logging
import warnings
from typing import Any

from langchain_core._api import LangChainDeprecationWarning

from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.utils.interactive_env import is_interactive_env


def __getattr__(name: str) -> Any:
    from langchain_community import embeddings

    # If not in interactive env, raise warning.
    if not is_interactive_env():
        warnings.warn(
            "Importing embeddings from langchain is deprecated. Importing from "
            "langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.embeddings import {name}`.\n\n"
            "To install langchain-community run `pip install -U langchain-community`.",
            category=LangChainDeprecationWarning,
        )

    return getattr(embeddings, name)


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
