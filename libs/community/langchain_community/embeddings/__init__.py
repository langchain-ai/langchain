"""**Embedding models**  are wrappers around embedding models
from different APIs and services.

**Embedding models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    Embeddings --> <name>Embeddings  # Examples: OpenAIEmbeddings, HuggingFaceEmbeddings
"""


import importlib
import logging
from typing import Any

_module_lookup = {
    "AlephAlphaAsymmetricSemanticEmbedding": "langchain_community.embeddings.aleph_alpha",  # noqa: E501
    "AlephAlphaSymmetricSemanticEmbedding": "langchain_community.embeddings.aleph_alpha",  # noqa: E501
    "AwaEmbeddings": "langchain_community.embeddings.awa",
    "AzureOpenAIEmbeddings": "langchain_community.embeddings.azure_openai",
    "BaichuanTextEmbeddings": "langchain_community.embeddings.baichuan",
    "BedrockEmbeddings": "langchain_community.embeddings.bedrock",
    "BookendEmbeddings": "langchain_community.embeddings.bookend",
    "ClarifaiEmbeddings": "langchain_community.embeddings.clarifai",
    "CohereEmbeddings": "langchain_community.embeddings.cohere",
    "DashScopeEmbeddings": "langchain_community.embeddings.dashscope",
    "DatabricksEmbeddings": "langchain_community.embeddings.databricks",
    "DeepInfraEmbeddings": "langchain_community.embeddings.deepinfra",
    "DeterministicFakeEmbedding": "langchain_community.embeddings.fake",
    "EdenAiEmbeddings": "langchain_community.embeddings.edenai",
    "ElasticsearchEmbeddings": "langchain_community.embeddings.elasticsearch",
    "EmbaasEmbeddings": "langchain_community.embeddings.embaas",
    "ErnieEmbeddings": "langchain_community.embeddings.ernie",
    "FakeEmbeddings": "langchain_community.embeddings.fake",
    "FastEmbedEmbeddings": "langchain_community.embeddings.fastembed",
    "GPT4AllEmbeddings": "langchain_community.embeddings.gpt4all",
    "GooglePalmEmbeddings": "langchain_community.embeddings.google_palm",
    "GradientEmbeddings": "langchain_community.embeddings.gradient_ai",
    "HuggingFaceBgeEmbeddings": "langchain_community.embeddings.huggingface",
    "HuggingFaceEmbeddings": "langchain_community.embeddings.huggingface",
    "HuggingFaceHubEmbeddings": "langchain_community.embeddings.huggingface_hub",
    "HuggingFaceInferenceAPIEmbeddings": "langchain_community.embeddings.huggingface",
    "HuggingFaceInstructEmbeddings": "langchain_community.embeddings.huggingface",
    "InfinityEmbeddings": "langchain_community.embeddings.infinity",
    "InfinityEmbeddingsLocal": "langchain_community.embeddings.infinity_local",
    "JavelinAIGatewayEmbeddings": "langchain_community.embeddings.javelin_ai_gateway",
    "JinaEmbeddings": "langchain_community.embeddings.jina",
    "JohnSnowLabsEmbeddings": "langchain_community.embeddings.johnsnowlabs",
    "LLMRailsEmbeddings": "langchain_community.embeddings.llm_rails",
    "LaserEmbeddings": "langchain_community.embeddings.laser",
    "LlamaCppEmbeddings": "langchain_community.embeddings.llamacpp",
    "LlamafileEmbeddings": "langchain_community.embeddings.llamafile",
    "LocalAIEmbeddings": "langchain_community.embeddings.localai",
    "MiniMaxEmbeddings": "langchain_community.embeddings.minimax",
    "MlflowAIGatewayEmbeddings": "langchain_community.embeddings.mlflow_gateway",
    "MlflowCohereEmbeddings": "langchain_community.embeddings.mlflow",
    "MlflowEmbeddings": "langchain_community.embeddings.mlflow",
    "ModelScopeEmbeddings": "langchain_community.embeddings.modelscope_hub",
    "MosaicMLInstructorEmbeddings": "langchain_community.embeddings.mosaicml",
    "NLPCloudEmbeddings": "langchain_community.embeddings.nlpcloud",
    "NeMoEmbeddings": "langchain_community.embeddings.nemo",
    "OCIGenAIEmbeddings": "langchain_community.embeddings.oci_generative_ai",
    "OctoAIEmbeddings": "langchain_community.embeddings.octoai_embeddings",
    "OllamaEmbeddings": "langchain_community.embeddings.ollama",
    "OpenAIEmbeddings": "langchain_community.embeddings.openai",
    "QianfanEmbeddingsEndpoint": "langchain_community.embeddings.baidu_qianfan_endpoint",  # noqa: E501
    "QuantizedBiEncoderEmbeddings": "langchain_community.embeddings.optimum_intel",
    "SagemakerEndpointEmbeddings": "langchain_community.embeddings.sagemaker_endpoint",
    "SelfHostedEmbeddings": "langchain_community.embeddings.self_hosted",
    "SelfHostedHuggingFaceEmbeddings": "langchain_community.embeddings.self_hosted_hugging_face",  # noqa: E501
    "SelfHostedHuggingFaceInstructEmbeddings": "langchain_community.embeddings.self_hosted_hugging_face",  # noqa: E501
    "SentenceTransformerEmbeddings": "langchain_community.embeddings.sentence_transformer",  # noqa: E501
    "SpacyEmbeddings": "langchain_community.embeddings.spacy_embeddings",
    "SparkLLMTextEmbeddings": "langchain_community.embeddings.sparkllm",
    "TensorflowHubEmbeddings": "langchain_community.embeddings.tensorflow_hub",
    "VertexAIEmbeddings": "langchain_community.embeddings.vertexai",
    "VolcanoEmbeddings": "langchain_community.embeddings.volcengine",
    "VoyageEmbeddings": "langchain_community.embeddings.voyageai",
    "XinferenceEmbeddings": "langchain_community.embeddings.xinference",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())

logger = logging.getLogger(__name__)


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
