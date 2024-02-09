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
from typing import Any, Callable, Dict, Type

from langchain_core.language_models.llms import BaseLLM


def _import_ai21() -> Any:
    from langchain_community.llms.ai21 import AI21

    return AI21


def _import_aleph_alpha() -> Any:
    from langchain_community.llms.aleph_alpha import AlephAlpha

    return AlephAlpha


def _import_amazon_api_gateway() -> Any:
    from langchain_community.llms.amazon_api_gateway import AmazonAPIGateway

    return AmazonAPIGateway


def _import_anthropic() -> Any:
    from langchain_community.llms.anthropic import Anthropic

    return Anthropic


def _import_anyscale() -> Any:
    from langchain_community.llms.anyscale import Anyscale

    return Anyscale


def _import_aphrodite() -> Any:
    from langchain_community.llms.aphrodite import Aphrodite

    return Aphrodite


def _import_arcee() -> Any:
    from langchain_community.llms.arcee import Arcee

    return Arcee


def _import_aviary() -> Any:
    from langchain_community.llms.aviary import Aviary

    return Aviary


def _import_azureml_endpoint() -> Any:
    from langchain_community.llms.azureml_endpoint import AzureMLOnlineEndpoint

    return AzureMLOnlineEndpoint


def _import_baichuan() -> Any:
    from langchain_community.llms.baichuan import BaichuanLLM

    return BaichuanLLM


def _import_baidu_qianfan_endpoint() -> Any:
    from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint

    return QianfanLLMEndpoint


def _import_bananadev() -> Any:
    from langchain_community.llms.bananadev import Banana

    return Banana


def _import_baseten() -> Any:
    from langchain_community.llms.baseten import Baseten

    return Baseten


def _import_beam() -> Any:
    from langchain_community.llms.beam import Beam

    return Beam


def _import_bedrock() -> Any:
    from langchain_community.llms.bedrock import Bedrock

    return Bedrock


def _import_bittensor() -> Any:
    from langchain_community.llms.bittensor import NIBittensorLLM

    return NIBittensorLLM


def _import_cerebriumai() -> Any:
    from langchain_community.llms.cerebriumai import CerebriumAI

    return CerebriumAI


def _import_chatglm() -> Any:
    from langchain_community.llms.chatglm import ChatGLM

    return ChatGLM


def _import_clarifai() -> Any:
    from langchain_community.llms.clarifai import Clarifai

    return Clarifai


def _import_cohere() -> Any:
    from langchain_community.llms.cohere import Cohere

    return Cohere


def _import_ctransformers() -> Any:
    from langchain_community.llms.ctransformers import CTransformers

    return CTransformers


def _import_ctranslate2() -> Any:
    from langchain_community.llms.ctranslate2 import CTranslate2

    return CTranslate2


def _import_databricks() -> Any:
    from langchain_community.llms.databricks import Databricks

    return Databricks


def _import_databricks_chat() -> Any:
    from langchain_community.chat_models.databricks import ChatDatabricks

    return ChatDatabricks


def _import_deepinfra() -> Any:
    from langchain_community.llms.deepinfra import DeepInfra

    return DeepInfra


def _import_deepsparse() -> Any:
    from langchain_community.llms.deepsparse import DeepSparse

    return DeepSparse


def _import_edenai() -> Any:
    from langchain_community.llms.edenai import EdenAI

    return EdenAI


def _import_fake() -> Any:
    from langchain_community.llms.fake import FakeListLLM

    return FakeListLLM


def _import_fireworks() -> Any:
    from langchain_community.llms.fireworks import Fireworks

    return Fireworks


def _import_forefrontai() -> Any:
    from langchain_community.llms.forefrontai import ForefrontAI

    return ForefrontAI


def _import_gigachat() -> Any:
    from langchain_community.llms.gigachat import GigaChat

    return GigaChat


def _import_google_palm() -> Any:
    from langchain_community.llms.google_palm import GooglePalm

    return GooglePalm


def _import_gooseai() -> Any:
    from langchain_community.llms.gooseai import GooseAI

    return GooseAI


def _import_gpt4all() -> Any:
    from langchain_community.llms.gpt4all import GPT4All

    return GPT4All


def _import_gradient_ai() -> Any:
    from langchain_community.llms.gradient_ai import GradientLLM

    return GradientLLM


def _import_huggingface_endpoint() -> Any:
    from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

    return HuggingFaceEndpoint


def _import_huggingface_hub() -> Any:
    from langchain_community.llms.huggingface_hub import HuggingFaceHub

    return HuggingFaceHub


def _import_huggingface_pipeline() -> Any:
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

    return HuggingFacePipeline


def _import_huggingface_text_gen_inference() -> Any:
    from langchain_community.llms.huggingface_text_gen_inference import (
        HuggingFaceTextGenInference,
    )

    return HuggingFaceTextGenInference


def _import_human() -> Any:
    from langchain_community.llms.human import HumanInputLLM

    return HumanInputLLM


def _import_javelin_ai_gateway() -> Any:
    from langchain_community.llms.javelin_ai_gateway import JavelinAIGateway

    return JavelinAIGateway


def _import_koboldai() -> Any:
    from langchain_community.llms.koboldai import KoboldApiLLM

    return KoboldApiLLM


def _import_konko() -> Any:
    from langchain_community.llms.konko import Konko

    return Konko


def _import_llamacpp() -> Any:
    from langchain_community.llms.llamacpp import LlamaCpp

    return LlamaCpp


def _import_manifest() -> Any:
    from langchain_community.llms.manifest import ManifestWrapper

    return ManifestWrapper


def _import_minimax() -> Any:
    from langchain_community.llms.minimax import Minimax

    return Minimax


def _import_mlflow() -> Any:
    from langchain_community.llms.mlflow import Mlflow

    return Mlflow


def _import_mlflow_chat() -> Any:
    from langchain_community.chat_models.mlflow import ChatMlflow

    return ChatMlflow


def _import_mlflow_ai_gateway() -> Any:
    from langchain_community.llms.mlflow_ai_gateway import MlflowAIGateway

    return MlflowAIGateway


def _import_modal() -> Any:
    from langchain_community.llms.modal import Modal

    return Modal


def _import_mosaicml() -> Any:
    from langchain_community.llms.mosaicml import MosaicML

    return MosaicML


def _import_nlpcloud() -> Any:
    from langchain_community.llms.nlpcloud import NLPCloud

    return NLPCloud


def _import_oci_md_tgi() -> Any:
    from langchain_community.llms.oci_data_science_model_deployment_endpoint import (
        OCIModelDeploymentTGI,
    )

    return OCIModelDeploymentTGI


def _import_oci_md_vllm() -> Any:
    from langchain_community.llms.oci_data_science_model_deployment_endpoint import (
        OCIModelDeploymentVLLM,
    )

    return OCIModelDeploymentVLLM


def _import_oci_gen_ai() -> Any:
    from langchain_community.llms.oci_generative_ai import OCIGenAI

    return OCIGenAI


def _import_octoai_endpoint() -> Any:
    from langchain_community.llms.octoai_endpoint import OctoAIEndpoint

    return OctoAIEndpoint


def _import_ollama() -> Any:
    from langchain_community.llms.ollama import Ollama

    return Ollama


def _import_opaqueprompts() -> Any:
    from langchain_community.llms.opaqueprompts import OpaquePrompts

    return OpaquePrompts


def _import_azure_openai() -> Any:
    from langchain_community.llms.openai import AzureOpenAI

    return AzureOpenAI


def _import_openai() -> Any:
    from langchain_community.llms.openai import OpenAI

    return OpenAI


def _import_openai_chat() -> Any:
    from langchain_community.llms.openai import OpenAIChat

    return OpenAIChat


def _import_openllm() -> Any:
    from langchain_community.llms.openllm import OpenLLM

    return OpenLLM


def _import_openlm() -> Any:
    from langchain_community.llms.openlm import OpenLM

    return OpenLM


def _import_pai_eas_endpoint() -> Any:
    from langchain_community.llms.pai_eas_endpoint import PaiEasEndpoint

    return PaiEasEndpoint


def _import_petals() -> Any:
    from langchain_community.llms.petals import Petals

    return Petals


def _import_pipelineai() -> Any:
    from langchain_community.llms.pipelineai import PipelineAI

    return PipelineAI


def _import_predibase() -> Any:
    from langchain_community.llms.predibase import Predibase

    return Predibase


def _import_predictionguard() -> Any:
    from langchain_community.llms.predictionguard import PredictionGuard

    return PredictionGuard


def _import_promptlayer() -> Any:
    from langchain_community.llms.promptlayer_openai import PromptLayerOpenAI

    return PromptLayerOpenAI


def _import_promptlayer_chat() -> Any:
    from langchain_community.llms.promptlayer_openai import PromptLayerOpenAIChat

    return PromptLayerOpenAIChat


def _import_replicate() -> Any:
    from langchain_community.llms.replicate import Replicate

    return Replicate


def _import_rwkv() -> Any:
    from langchain_community.llms.rwkv import RWKV

    return RWKV


def _import_sagemaker_endpoint() -> Any:
    from langchain_community.llms.sagemaker_endpoint import SagemakerEndpoint

    return SagemakerEndpoint


def _import_self_hosted() -> Any:
    from langchain_community.llms.self_hosted import SelfHostedPipeline

    return SelfHostedPipeline


def _import_self_hosted_hugging_face() -> Any:
    from langchain_community.llms.self_hosted_hugging_face import (
        SelfHostedHuggingFaceLLM,
    )

    return SelfHostedHuggingFaceLLM


def _import_stochasticai() -> Any:
    from langchain_community.llms.stochasticai import StochasticAI

    return StochasticAI


def _import_symblai_nebula() -> Any:
    from langchain_community.llms.symblai_nebula import Nebula

    return Nebula


def _import_textgen() -> Any:
    from langchain_community.llms.textgen import TextGen

    return TextGen


def _import_titan_takeoff() -> Any:
    from langchain_community.llms.titan_takeoff import TitanTakeoff

    return TitanTakeoff


def _import_titan_takeoff_pro() -> Any:
    from langchain_community.llms.titan_takeoff_pro import TitanTakeoffPro

    return TitanTakeoffPro


def _import_together() -> Any:
    from langchain_community.llms.together import Together

    return Together


def _import_tongyi() -> Any:
    from langchain_community.llms.tongyi import Tongyi

    return Tongyi


def _import_vertex() -> Any:
    from langchain_community.llms.vertexai import VertexAI

    return VertexAI


def _import_vertex_model_garden() -> Any:
    from langchain_community.llms.vertexai import VertexAIModelGarden

    return VertexAIModelGarden


def _import_vllm() -> Any:
    from langchain_community.llms.vllm import VLLM

    return VLLM


def _import_vllm_openai() -> Any:
    from langchain_community.llms.vllm import VLLMOpenAI

    return VLLMOpenAI


def _import_watsonxllm() -> Any:
    from langchain_community.llms.watsonxllm import WatsonxLLM

    return WatsonxLLM


def _import_writer() -> Any:
    from langchain_community.llms.writer import Writer

    return Writer


def _import_xinference() -> Any:
    from langchain_community.llms.xinference import Xinference

    return Xinference


def _import_yandex_gpt() -> Any:
    from langchain_community.llms.yandex import YandexGPT

    return YandexGPT


def _import_volcengine_maas() -> Any:
    from langchain_community.llms.volcengine_maas import VolcEngineMaasLLM

    return VolcEngineMaasLLM


def __getattr__(name: str) -> Any:
    if name == "AI21":
        return _import_ai21()
    elif name == "AlephAlpha":
        return _import_aleph_alpha()
    elif name == "AmazonAPIGateway":
        return _import_amazon_api_gateway()
    elif name == "Anthropic":
        return _import_anthropic()
    elif name == "Anyscale":
        return _import_anyscale()
    elif name == "Aphrodite":
        return _import_aphrodite()
    elif name == "Arcee":
        return _import_arcee()
    elif name == "Aviary":
        return _import_aviary()
    elif name == "AzureMLOnlineEndpoint":
        return _import_azureml_endpoint()
    elif name == "Baichuan":
        return _import_baichuan()
    elif name == "QianfanLLMEndpoint":
        return _import_baidu_qianfan_endpoint()
    elif name == "Banana":
        return _import_bananadev()
    elif name == "Baseten":
        return _import_baseten()
    elif name == "Beam":
        return _import_beam()
    elif name == "Bedrock":
        return _import_bedrock()
    elif name == "NIBittensorLLM":
        return _import_bittensor()
    elif name == "CerebriumAI":
        return _import_cerebriumai()
    elif name == "ChatGLM":
        return _import_chatglm()
    elif name == "Clarifai":
        return _import_clarifai()
    elif name == "Cohere":
        return _import_cohere()
    elif name == "CTransformers":
        return _import_ctransformers()
    elif name == "CTranslate2":
        return _import_ctranslate2()
    elif name == "Databricks":
        return _import_databricks()
    elif name == "DeepInfra":
        return _import_deepinfra()
    elif name == "DeepSparse":
        return _import_deepsparse()
    elif name == "EdenAI":
        return _import_edenai()
    elif name == "FakeListLLM":
        return _import_fake()
    elif name == "Fireworks":
        return _import_fireworks()
    elif name == "ForefrontAI":
        return _import_forefrontai()
    elif name == "GigaChat":
        return _import_gigachat()
    elif name == "GooglePalm":
        return _import_google_palm()
    elif name == "GooseAI":
        return _import_gooseai()
    elif name == "GPT4All":
        return _import_gpt4all()
    elif name == "GradientLLM":
        return _import_gradient_ai()
    elif name == "HuggingFaceEndpoint":
        return _import_huggingface_endpoint()
    elif name == "HuggingFaceHub":
        return _import_huggingface_hub()
    elif name == "HuggingFacePipeline":
        return _import_huggingface_pipeline()
    elif name == "HuggingFaceTextGenInference":
        return _import_huggingface_text_gen_inference()
    elif name == "HumanInputLLM":
        return _import_human()
    elif name == "JavelinAIGateway":
        return _import_javelin_ai_gateway()
    elif name == "KoboldApiLLM":
        return _import_koboldai()
    elif name == "Konko":
        return _import_konko()
    elif name == "LlamaCpp":
        return _import_llamacpp()
    elif name == "ManifestWrapper":
        return _import_manifest()
    elif name == "Minimax":
        return _import_minimax()
    elif name == "Mlflow":
        return _import_mlflow()
    elif name == "MlflowAIGateway":
        return _import_mlflow_ai_gateway()
    elif name == "Modal":
        return _import_modal()
    elif name == "MosaicML":
        return _import_mosaicml()
    elif name == "NLPCloud":
        return _import_nlpcloud()
    elif name == "OCIModelDeploymentTGI":
        return _import_oci_md_tgi()
    elif name == "OCIModelDeploymentVLLM":
        return _import_oci_md_vllm()
    elif name == "OCIGenAI":
        return _import_oci_gen_ai()
    elif name == "OctoAIEndpoint":
        return _import_octoai_endpoint()
    elif name == "Ollama":
        return _import_ollama()
    elif name == "OpaquePrompts":
        return _import_opaqueprompts()
    elif name == "AzureOpenAI":
        return _import_azure_openai()
    elif name == "OpenAI":
        return _import_openai()
    elif name == "OpenAIChat":
        return _import_openai_chat()
    elif name == "OpenLLM":
        return _import_openllm()
    elif name == "OpenLM":
        return _import_openlm()
    elif name == "PaiEasEndpoint":
        return _import_pai_eas_endpoint()
    elif name == "Petals":
        return _import_petals()
    elif name == "PipelineAI":
        return _import_pipelineai()
    elif name == "Predibase":
        return _import_predibase()
    elif name == "PredictionGuard":
        return _import_predictionguard()
    elif name == "PromptLayerOpenAI":
        return _import_promptlayer()
    elif name == "PromptLayerOpenAIChat":
        return _import_promptlayer_chat()
    elif name == "Replicate":
        return _import_replicate()
    elif name == "RWKV":
        return _import_rwkv()
    elif name == "SagemakerEndpoint":
        return _import_sagemaker_endpoint()
    elif name == "SelfHostedPipeline":
        return _import_self_hosted()
    elif name == "SelfHostedHuggingFaceLLM":
        return _import_self_hosted_hugging_face()
    elif name == "StochasticAI":
        return _import_stochasticai()
    elif name == "Nebula":
        return _import_symblai_nebula()
    elif name == "TextGen":
        return _import_textgen()
    elif name == "TitanTakeoff":
        return _import_titan_takeoff()
    elif name == "TitanTakeoffPro":
        return _import_titan_takeoff_pro()
    elif name == "Together":
        return _import_together()
    elif name == "Tongyi":
        return _import_tongyi()
    elif name == "VertexAI":
        return _import_vertex()
    elif name == "VertexAIModelGarden":
        return _import_vertex_model_garden()
    elif name == "VLLM":
        return _import_vllm()
    elif name == "VLLMOpenAI":
        return _import_vllm_openai()
    elif name == "WatsonxLLM":
        return _import_watsonxllm()
    elif name == "Writer":
        return _import_writer()
    elif name == "Xinference":
        return _import_xinference()
    elif name == "YandexGPT":
        return _import_yandex_gpt()
    elif name == "VolcEngineMaasLLM":
        return _import_volcengine_maas()
    elif name == "type_to_cls_dict":
        # for backwards compatibility
        type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
            k: v() for k, v in get_type_to_cls_dict().items()
        }
        return type_to_cls_dict
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = [
    "AI21",
    "AlephAlpha",
    "AmazonAPIGateway",
    "Anthropic",
    "Anyscale",
    "Aphrodite",
    "Arcee",
    "Aviary",
    "AzureMLOnlineEndpoint",
    "AzureOpenAI",
    "Banana",
    "Baseten",
    "Beam",
    "Bedrock",
    "CTransformers",
    "CTranslate2",
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
    "ForefrontAI",
    "GigaChat",
    "GPT4All",
    "GooglePalm",
    "GooseAI",
    "GradientLLM",
    "HuggingFaceEndpoint",
    "HuggingFaceHub",
    "HuggingFacePipeline",
    "HuggingFaceTextGenInference",
    "HumanInputLLM",
    "KoboldApiLLM",
    "Konko",
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
    "OCIModelDeploymentTGI",
    "OCIModelDeploymentVLLM",
    "OCIGenAI",
    "Ollama",
    "OpenAI",
    "OpenAIChat",
    "OpenLLM",
    "OpenLM",
    "PaiEasEndpoint",
    "Petals",
    "PipelineAI",
    "Predibase",
    "PredictionGuard",
    "PromptLayerOpenAI",
    "PromptLayerOpenAIChat",
    "OpaquePrompts",
    "RWKV",
    "Replicate",
    "SagemakerEndpoint",
    "SelfHostedHuggingFaceLLM",
    "SelfHostedPipeline",
    "StochasticAI",
    "TitanTakeoff",
    "TitanTakeoffPro",
    "Tongyi",
    "VertexAI",
    "VertexAIModelGarden",
    "VLLM",
    "VLLMOpenAI",
    "WatsonxLLM",
    "Writer",
    "OctoAIEndpoint",
    "Xinference",
    "JavelinAIGateway",
    "QianfanLLMEndpoint",
    "YandexGPT",
    "VolcEngineMaasLLM",
]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[BaseLLM]]]:
    return {
        "ai21": _import_ai21,
        "aleph_alpha": _import_aleph_alpha,
        "amazon_api_gateway": _import_amazon_api_gateway,
        "amazon_bedrock": _import_bedrock,
        "anthropic": _import_anthropic,
        "anyscale": _import_anyscale,
        "arcee": _import_arcee,
        "aviary": _import_aviary,
        "azure": _import_azure_openai,
        "azureml_endpoint": _import_azureml_endpoint,
        "bananadev": _import_bananadev,
        "baseten": _import_baseten,
        "beam": _import_beam,
        "cerebriumai": _import_cerebriumai,
        "chat_glm": _import_chatglm,
        "clarifai": _import_clarifai,
        "cohere": _import_cohere,
        "ctransformers": _import_ctransformers,
        "ctranslate2": _import_ctranslate2,
        "databricks": _import_databricks,
        "databricks-chat": _import_databricks_chat,
        "deepinfra": _import_deepinfra,
        "deepsparse": _import_deepsparse,
        "edenai": _import_edenai,
        "fake-list": _import_fake,
        "forefrontai": _import_forefrontai,
        "giga-chat-model": _import_gigachat,
        "google_palm": _import_google_palm,
        "gooseai": _import_gooseai,
        "gradient": _import_gradient_ai,
        "gpt4all": _import_gpt4all,
        "huggingface_endpoint": _import_huggingface_endpoint,
        "huggingface_hub": _import_huggingface_hub,
        "huggingface_pipeline": _import_huggingface_pipeline,
        "huggingface_textgen_inference": _import_huggingface_text_gen_inference,
        "human-input": _import_human,
        "koboldai": _import_koboldai,
        "konko": _import_konko,
        "llamacpp": _import_llamacpp,
        "textgen": _import_textgen,
        "minimax": _import_minimax,
        "mlflow": _import_mlflow,
        "mlflow-chat": _import_mlflow_chat,
        "mlflow-ai-gateway": _import_mlflow_ai_gateway,
        "modal": _import_modal,
        "mosaic": _import_mosaicml,
        "nebula": _import_symblai_nebula,
        "nibittensor": _import_bittensor,
        "nlpcloud": _import_nlpcloud,
        "oci_model_deployment_tgi_endpoint": _import_oci_md_tgi,
        "oci_model_deployment_vllm_endpoint": _import_oci_md_vllm,
        "oci_generative_ai": _import_oci_gen_ai,
        "ollama": _import_ollama,
        "openai": _import_openai,
        "openlm": _import_openlm,
        "pai_eas_endpoint": _import_pai_eas_endpoint,
        "petals": _import_petals,
        "pipelineai": _import_pipelineai,
        "predibase": _import_predibase,
        "opaqueprompts": _import_opaqueprompts,
        "replicate": _import_replicate,
        "rwkv": _import_rwkv,
        "sagemaker_endpoint": _import_sagemaker_endpoint,
        "self_hosted": _import_self_hosted,
        "self_hosted_hugging_face": _import_self_hosted_hugging_face,
        "stochasticai": _import_stochasticai,
        "together": _import_together,
        "tongyi": _import_tongyi,
        "titan_takeoff": _import_titan_takeoff,
        "titan_takeoff_pro": _import_titan_takeoff_pro,
        "vertexai": _import_vertex,
        "vertexai_model_garden": _import_vertex_model_garden,
        "openllm": _import_openllm,
        "openllm_client": _import_openllm,
        "vllm": _import_vllm,
        "vllm_openai": _import_vllm_openai,
        "watsonxllm": _import_watsonxllm,
        "writer": _import_writer,
        "xinference": _import_xinference,
        "javelin-ai-gateway": _import_javelin_ai_gateway,
        "qianfan_endpoint": _import_baidu_qianfan_endpoint,
        "yandex_gpt": _import_yandex_gpt,
        "VolcEngineMaasLLM": _import_volcengine_maas,
    }
