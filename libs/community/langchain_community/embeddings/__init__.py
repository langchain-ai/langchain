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
        AlephAlphaAsymmetricSemanticEmbedding,
        AlephAlphaSymmetricSemanticEmbedding,
    )
    from langchain_community.embeddings.anyscale import (
        AnyscaleEmbeddings,
    )
    from langchain_community.embeddings.ascend import (
        AscendEmbeddings,
    )
    from langchain_community.embeddings.awa import (
        AwaEmbeddings,
    )
    from langchain_community.embeddings.azure_openai import (
        AzureOpenAIEmbeddings,
    )
    from langchain_community.embeddings.baichuan import (
        BaichuanTextEmbeddings,
    )
    from langchain_community.embeddings.baidu_qianfan_endpoint import (
        QianfanEmbeddingsEndpoint,
    )
    from langchain_community.embeddings.bedrock import (
        BedrockEmbeddings,
    )
    from langchain_community.embeddings.bookend import (
        BookendEmbeddings,
    )
    from langchain_community.embeddings.clarifai import (
        ClarifaiEmbeddings,
    )
    from langchain_community.embeddings.clova import (
        ClovaEmbeddings,
    )
    from langchain_community.embeddings.cohere import (
        CohereEmbeddings,
    )
    from langchain_community.embeddings.dashscope import (
        DashScopeEmbeddings,
    )
    from langchain_community.embeddings.databricks import (
        DatabricksEmbeddings,
    )
    from langchain_community.embeddings.deepinfra import (
        DeepInfraEmbeddings,
    )
    from langchain_community.embeddings.edenai import (
        EdenAiEmbeddings,
    )
    from langchain_community.embeddings.elasticsearch import (
        ElasticsearchEmbeddings,
    )
    from langchain_community.embeddings.embaas import (
        EmbaasEmbeddings,
    )
    from langchain_community.embeddings.ernie import (
        ErnieEmbeddings,
    )
    from langchain_community.embeddings.fake import (
        DeterministicFakeEmbedding,
        FakeEmbeddings,
    )
    from langchain_community.embeddings.fastembed import (
        FastEmbedEmbeddings,
    )
    from langchain_community.embeddings.gigachat import (
        GigaChatEmbeddings,
    )
    from langchain_community.embeddings.google_palm import (
        GooglePalmEmbeddings,
    )
    from langchain_community.embeddings.gpt4all import (
        GPT4AllEmbeddings,
    )
    from langchain_community.embeddings.gradient_ai import (
        GradientEmbeddings,
    )
    from langchain_community.embeddings.huggingface import (
        HuggingFaceBgeEmbeddings,
        HuggingFaceEmbeddings,
        HuggingFaceInferenceAPIEmbeddings,
        HuggingFaceInstructEmbeddings,
    )
    from langchain_community.embeddings.huggingface_hub import (
        HuggingFaceHubEmbeddings,
    )
    from langchain_community.embeddings.infinity import (
        InfinityEmbeddings,
    )
    from langchain_community.embeddings.infinity_local import (
        InfinityEmbeddingsLocal,
    )
    from langchain_community.embeddings.ipex_llm import IpexLLMBgeEmbeddings
    from langchain_community.embeddings.itrex import (
        QuantizedBgeEmbeddings,
    )
    from langchain_community.embeddings.javelin_ai_gateway import (
        JavelinAIGatewayEmbeddings,
    )
    from langchain_community.embeddings.jina import (
        JinaEmbeddings,
    )
    from langchain_community.embeddings.johnsnowlabs import (
        JohnSnowLabsEmbeddings,
    )
    from langchain_community.embeddings.laser import (
        LaserEmbeddings,
    )
    from langchain_community.embeddings.llamacpp import (
        LlamaCppEmbeddings,
    )
    from langchain_community.embeddings.llamafile import (
        LlamafileEmbeddings,
    )
    from langchain_community.embeddings.llm_rails import (
        LLMRailsEmbeddings,
    )
    from langchain_community.embeddings.localai import (
        LocalAIEmbeddings,
    )
    from langchain_community.embeddings.minimax import (
        MiniMaxEmbeddings,
    )
    from langchain_community.embeddings.mlflow import (
        MlflowCohereEmbeddings,
        MlflowEmbeddings,
    )
    from langchain_community.embeddings.mlflow_gateway import (
        MlflowAIGatewayEmbeddings,
    )
    from langchain_community.embeddings.modelscope_hub import (
        ModelScopeEmbeddings,
    )
    from langchain_community.embeddings.mosaicml import (
        MosaicMLInstructorEmbeddings,
    )
    from langchain_community.embeddings.nemo import (
        NeMoEmbeddings,
    )
    from langchain_community.embeddings.nlpcloud import (
        NLPCloudEmbeddings,
    )
    from langchain_community.embeddings.oci_generative_ai import (
        OCIGenAIEmbeddings,
    )
    from langchain_community.embeddings.octoai_embeddings import (
        OctoAIEmbeddings,
    )
    from langchain_community.embeddings.ollama import (
        OllamaEmbeddings,
    )
    from langchain_community.embeddings.openai import (
        OpenAIEmbeddings,
    )
    from langchain_community.embeddings.openvino import (
        OpenVINOBgeEmbeddings,
        OpenVINOEmbeddings,
    )
    from langchain_community.embeddings.optimum_intel import (
        QuantizedBiEncoderEmbeddings,
    )
    from langchain_community.embeddings.oracleai import (
        OracleEmbeddings,
    )
    from langchain_community.embeddings.ovhcloud import (
        OVHCloudEmbeddings,
    )
    from langchain_community.embeddings.premai import (
        PremAIEmbeddings,
    )
    from langchain_community.embeddings.sagemaker_endpoint import (
        SagemakerEndpointEmbeddings,
    )
    from langchain_community.embeddings.sambanova import (
        SambaStudioEmbeddings,
    )
    from langchain_community.embeddings.self_hosted import (
        SelfHostedEmbeddings,
    )
    from langchain_community.embeddings.self_hosted_hugging_face import (
        SelfHostedHuggingFaceEmbeddings,
        SelfHostedHuggingFaceInstructEmbeddings,
    )
    from langchain_community.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddings,
    )
    from langchain_community.embeddings.solar import (
        SolarEmbeddings,
    )
    from langchain_community.embeddings.spacy_embeddings import (
        SpacyEmbeddings,
    )
    from langchain_community.embeddings.sparkllm import (
        SparkLLMTextEmbeddings,
    )
    from langchain_community.embeddings.tensorflow_hub import (
        TensorflowHubEmbeddings,
    )
    from langchain_community.embeddings.titan_takeoff import (
        TitanTakeoffEmbed,
    )
    from langchain_community.embeddings.vertexai import (
        VertexAIEmbeddings,
    )
    from langchain_community.embeddings.volcengine import (
        VolcanoEmbeddings,
    )
    from langchain_community.embeddings.voyageai import (
        VoyageEmbeddings,
    )
    from langchain_community.embeddings.xinference import (
        XinferenceEmbeddings,
    )
    from langchain_community.embeddings.yandex import (
        YandexGPTEmbeddings,
    )
    from langchain_community.embeddings.zhipuai import (
        ZhipuAIEmbeddings,
    )

__all__ = [
    "AlephAlphaAsymmetricSemanticEmbedding",
    "AlephAlphaSymmetricSemanticEmbedding",
    "AnyscaleEmbeddings",
    "AscendEmbeddings",
    "AwaEmbeddings",
    "AzureOpenAIEmbeddings",
    "BaichuanTextEmbeddings",
    "BedrockEmbeddings",
    "BookendEmbeddings",
    "ClarifaiEmbeddings",
    "ClovaEmbeddings",
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
    "IpexLLMBgeEmbeddings",
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
    "OracleEmbeddings",
    "OVHCloudEmbeddings",
    "PremAIEmbeddings",
    "QianfanEmbeddingsEndpoint",
    "QuantizedBgeEmbeddings",
    "QuantizedBiEncoderEmbeddings",
    "SagemakerEndpointEmbeddings",
    "SambaStudioEmbeddings",
    "SelfHostedEmbeddings",
    "SelfHostedHuggingFaceEmbeddings",
    "SelfHostedHuggingFaceInstructEmbeddings",
    "SentenceTransformerEmbeddings",
    "SolarEmbeddings",
    "SpacyEmbeddings",
    "SparkLLMTextEmbeddings",
    "TensorflowHubEmbeddings",
    "TitanTakeoffEmbed",
    "VertexAIEmbeddings",
    "VolcanoEmbeddings",
    "VoyageEmbeddings",
    "XinferenceEmbeddings",
    "YandexGPTEmbeddings",
    "ZhipuAIEmbeddings",
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
    "ClovaEmbeddings": "langchain_community.embeddings.clova",
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
    "IpexLLMBgeEmbeddings": "langchain_community.embeddings.ipex_llm",
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
    "OracleEmbeddings": "langchain_community.embeddings.oracleai",
    "OVHCloudEmbeddings": "langchain_community.embeddings.ovhcloud",
    "SagemakerEndpointEmbeddings": "langchain_community.embeddings.sagemaker_endpoint",
    "SambaStudioEmbeddings": "langchain_community.embeddings.sambanova",
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
    "AscendEmbeddings": "langchain_community.embeddings.ascend",
    "ZhipuAIEmbeddings": "langchain_community.embeddings.zhipuai",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


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
