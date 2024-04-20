"""**Embedding models**  are wrappers around embedding models
from different APIs and services.

**Embedding models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    Embeddings --> <name>Embeddings  # Examples: OpenAIEmbeddings, HuggingFaceEmbeddings
"""

import importlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.embeddings.aleph_alpha import (
        AlephAlphaAsymmetricSemanticEmbedding,  # noqa: F401
        AlephAlphaSymmetricSemanticEmbedding,  # noqa: F401
    )
    from langchain_community.embeddings.anyscale import (
        AnyscaleEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.awa import (
        AwaEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.azure_openai import (
        AzureOpenAIEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.baichuan import (
        BaichuanTextEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.baidu_qianfan_endpoint import (
        QianfanEmbeddingsEndpoint,  # noqa: F401
    )
    from langchain_community.embeddings.bedrock import (
        BedrockEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.bookend import (
        BookendEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.clarifai import (
        ClarifaiEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.cohere import (
        CohereEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.dashscope import (
        DashScopeEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.databricks import (
        DatabricksEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.deepinfra import (
        DeepInfraEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.edenai import (
        EdenAiEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.elasticsearch import (
        ElasticsearchEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.embaas import (
        EmbaasEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.ernie import (
        ErnieEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.fake import (
        DeterministicFakeEmbedding,  # noqa: F401
        FakeEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.fastembed import (
        FastEmbedEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.gigachat import (
        GigaChatEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.google_palm import (
        GooglePalmEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.gpt4all import (
        GPT4AllEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.gradient_ai import (
        GradientEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.huggingface import (
        HuggingFaceBgeEmbeddings,  # noqa: F401
        HuggingFaceEmbeddings,  # noqa: F401
        HuggingFaceInferenceAPIEmbeddings,  # noqa: F401
        HuggingFaceInstructEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.huggingface_hub import (
        HuggingFaceHubEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.infinity import (
        InfinityEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.infinity_local import (
        InfinityEmbeddingsLocal,  # noqa: F401
    )
    from langchain_community.embeddings.itrex import (
        QuantizedBgeEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.javelin_ai_gateway import (
        JavelinAIGatewayEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.jina import (
        JinaEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.johnsnowlabs import (
        JohnSnowLabsEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.laser import (
        LaserEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.llamacpp import (
        LlamaCppEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.llamafile import (
        LlamafileEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.llm_rails import (
        LLMRailsEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.localai import (
        LocalAIEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.minimax import (
        MiniMaxEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.mlflow import (
        MlflowCohereEmbeddings,  # noqa: F401
        MlflowEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.mlflow_gateway import (
        MlflowAIGatewayEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.modelscope_hub import (
        ModelScopeEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.mosaicml import (
        MosaicMLInstructorEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.nemo import (
        NeMoEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.nlpcloud import (
        NLPCloudEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.oci_generative_ai import (
        OCIGenAIEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.octoai_embeddings import (
        OctoAIEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.ollama import (
        OllamaEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.openai import (
        OpenAIEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.openvino import (
        OpenVINOBgeEmbeddings,  # noqa: F401
        OpenVINOEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.optimum_intel import (
        QuantizedBiEncoderEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.premai import (
        PremAIEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.sagemaker_endpoint import (
        SagemakerEndpointEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.self_hosted import (
        SelfHostedEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.self_hosted_hugging_face import (
        SelfHostedHuggingFaceEmbeddings,  # noqa: F401
        SelfHostedHuggingFaceInstructEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.solar import (
        SolarEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.spacy_embeddings import (
        SpacyEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.sparkllm import (
        SparkLLMTextEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.tensorflow_hub import (
        TensorflowHubEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.vertexai import (
        VertexAIEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.volcengine import (
        VolcanoEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.voyageai import (
        VoyageEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.xinference import (
        XinferenceEmbeddings,  # noqa: F401
    )
    from langchain_community.embeddings.yandex import (
        YandexGPTEmbeddings,  # noqa: F401
    )

__all__ = [
    "AlephAlphaAsymmetricSemanticEmbedding",
    "AlephAlphaSymmetricSemanticEmbedding",
    "AnyscaleEmbeddings",
    "AwaEmbeddings",
    "AzureOpenAIEmbeddings",
    "BaichuanTextEmbeddings",
    "BedrockEmbeddings",
    "BookendEmbeddings",
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
    "GigaChatEmbeddings",
    "GooglePalmEmbeddings",
    "GradientEmbeddings",
    "HuggingFaceBgeEmbeddings",
    "HuggingFaceEmbeddings",
    "HuggingFaceHubEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
    "HuggingFaceInstructEmbeddings",
    "InfinityEmbeddings",
    "InfinityEmbeddingsLocal",
    "JavelinAIGatewayEmbeddings",
    "JinaEmbeddings",
    "JohnSnowLabsEmbeddings",
    "LLMRailsEmbeddings",
    "LaserEmbeddings",
    "LlamaCppEmbeddings",
    "LlamafileEmbeddings",
    "LocalAIEmbeddings",
    "MiniMaxEmbeddings",
    "MlflowAIGatewayEmbeddings",
    "MlflowCohereEmbeddings",
    "MlflowEmbeddings",
    "ModelScopeEmbeddings",
    "MosaicMLInstructorEmbeddings",
    "NLPCloudEmbeddings",
    "NeMoEmbeddings",
    "OCIGenAIEmbeddings",
    "OctoAIEmbeddings",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    "OpenVINOBgeEmbeddings",
    "OpenVINOEmbeddings",
    "PremAIEmbeddings",
    "QianfanEmbeddingsEndpoint",
    "QuantizedBgeEmbeddings",
    "QuantizedBiEncoderEmbeddings",
    "SagemakerEndpointEmbeddings",
    "SelfHostedEmbeddings",
    "SelfHostedHuggingFaceEmbeddings",
    "SelfHostedHuggingFaceInstructEmbeddings",
    "SentenceTransformerEmbeddings",
    "SolarEmbeddings",
    "SpacyEmbeddings",
    "SparkLLMTextEmbeddings",
    "TensorflowHubEmbeddings",
    "VertexAIEmbeddings",
    "VolcanoEmbeddings",
    "VoyageEmbeddings",
    "XinferenceEmbeddings",
    "YandexGPTEmbeddings",
]

_module_lookup = {
    "AlephAlphaAsymmetricSemanticEmbedding": "langchain_community.embeddings.aleph_alpha",  # noqa: E501
    "AlephAlphaSymmetricSemanticEmbedding": "langchain_community.embeddings.aleph_alpha",  # noqa: E501
    "AnyscaleEmbeddings": "langchain_community.embeddings.anyscale",
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
    "GigaChatEmbeddings": "langchain_community.embeddings.gigachat",
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
    "OpenVINOEmbeddings": "langchain_community.embeddings.openvino",
    "OpenVINOBgeEmbeddings": "langchain_community.embeddings.openvino",
    "QianfanEmbeddingsEndpoint": "langchain_community.embeddings.baidu_qianfan_endpoint",  # noqa: E501
    "QuantizedBgeEmbeddings": "langchain_community.embeddings.itrex",
    "QuantizedBiEncoderEmbeddings": "langchain_community.embeddings.optimum_intel",
    "SagemakerEndpointEmbeddings": "langchain_community.embeddings.sagemaker_endpoint",
    "SelfHostedEmbeddings": "langchain_community.embeddings.self_hosted",
    "SelfHostedHuggingFaceEmbeddings": "langchain_community.embeddings.self_hosted_hugging_face",  # noqa: E501
    "SelfHostedHuggingFaceInstructEmbeddings": "langchain_community.embeddings.self_hosted_hugging_face",  # noqa: E501
    "SentenceTransformerEmbeddings": "langchain_community.embeddings.sentence_transformer",  # noqa: E501
    "SolarEmbeddings": "langchain_community.embeddings.solar",
    "SpacyEmbeddings": "langchain_community.embeddings.spacy_embeddings",
    "SparkLLMTextEmbeddings": "langchain_community.embeddings.sparkllm",
    "TensorflowHubEmbeddings": "langchain_community.embeddings.tensorflow_hub",
    "VertexAIEmbeddings": "langchain_community.embeddings.vertexai",
    "VolcanoEmbeddings": "langchain_community.embeddings.volcengine",
    "VoyageEmbeddings": "langchain_community.embeddings.voyageai",
    "XinferenceEmbeddings": "langchain_community.embeddings.xinference",
    "TitanTakeoffEmbed": "langchain_community.embeddings.titan_takeoff",
    "PremAIEmbeddings": "langchain_community.embeddings.premai",
    "YandexGPTEmbeddings": "langchain_community.embeddings.yandex",
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
