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

from langchain.llms.base import BaseLLM


def _import_ai21() -> Any:
    from langchain.llms.ai21 import AI21

    return AI21


def _import_aleph_alpha() -> Any:
    from langchain.llms.aleph_alpha import AlephAlpha

    return AlephAlpha


def _import_amazon_api_gateway() -> Any:
    from langchain.llms.amazon_api_gateway import AmazonAPIGateway

    return AmazonAPIGateway


def _import_anthropic() -> Any:
    from langchain.llms.anthropic import Anthropic

    return Anthropic


def _import_anyscale() -> Any:
    from langchain.llms.anyscale import Anyscale

    return Anyscale


def _import_arcee() -> Any:
    from langchain.llms.arcee import Arcee

    return Arcee


def _import_aviary() -> Any:
    from langchain.llms.aviary import Aviary

    return Aviary


def _import_azureml_endpoint() -> Any:
    from langchain.llms.azureml_endpoint import AzureMLOnlineEndpoint

    return AzureMLOnlineEndpoint


def _import_baidu_qianfan_endpoint() -> Any:
    from langchain.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint

    return QianfanLLMEndpoint


def _import_bananadev() -> Any:
    from langchain.llms.bananadev import Banana

    return Banana


def _import_baseten() -> Any:
    from langchain.llms.baseten import Baseten

    return Baseten


def _import_beam() -> Any:
    from langchain.llms.beam import Beam

    return Beam


def _import_bedrock() -> Any:
    from langchain.llms.bedrock import Bedrock

    return Bedrock


def _import_bittensor() -> Any:
    from langchain.llms.bittensor import NIBittensorLLM

    return NIBittensorLLM


def _import_cerebriumai() -> Any:
    from langchain.llms.cerebriumai import CerebriumAI

    return CerebriumAI


def _import_chatglm() -> Any:
    from langchain.llms.chatglm import ChatGLM

    return ChatGLM


def _import_clarifai() -> Any:
    from langchain.llms.clarifai import Clarifai

    return Clarifai


def _import_cohere() -> Any:
    from langchain.llms.cohere import Cohere

    return Cohere


def _import_ctransformers() -> Any:
    from langchain.llms.ctransformers import CTransformers

    return CTransformers


def _import_ctranslate2() -> Any:
    from langchain.llms.ctranslate2 import CTranslate2

    return CTranslate2


def _import_databricks() -> Any:
    from langchain.llms.databricks import Databricks

    return Databricks


def _import_databricks_chat() -> Any:
    from langchain.chat_models.databricks import ChatDatabricks

    return ChatDatabricks


def _import_deepinfra() -> Any:
    from langchain.llms.deepinfra import DeepInfra

    return DeepInfra


def _import_deepsparse() -> Any:
    from langchain.llms.deepsparse import DeepSparse

    return DeepSparse


def _import_edenai() -> Any:
    from langchain.llms.edenai import EdenAI

    return EdenAI


def _import_fake() -> Any:
    from langchain.llms.fake import FakeListLLM

    return FakeListLLM


def _import_fireworks() -> Any:
    from langchain.llms.fireworks import Fireworks

    return Fireworks


def _import_forefrontai() -> Any:
    from langchain.llms.forefrontai import ForefrontAI

    return ForefrontAI


def _import_gigachat() -> Any:
    from langchain.llms.gigachat import GigaChat

    return GigaChat


def _import_google_palm() -> Any:
    from langchain.llms.google_palm import GooglePalm

    return GooglePalm


def _import_gooseai() -> Any:
    from langchain.llms.gooseai import GooseAI

    return GooseAI


def _import_gpt4all() -> Any:
    from langchain.llms.gpt4all import GPT4All

    return GPT4All


def _import_gradient_ai() -> Any:
    from langchain.llms.gradient_ai import GradientLLM

    return GradientLLM


def _import_huggingface_endpoint() -> Any:
    from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint

    return HuggingFaceEndpoint


def _import_huggingface_hub() -> Any:
    from langchain.llms.huggingface_hub import HuggingFaceHub

    return HuggingFaceHub


def _import_huggingface_pipeline() -> Any:
    from langchain.llms.huggingface_pipeline import HuggingFacePipeline

    return HuggingFacePipeline


def _import_huggingface_text_gen_inference() -> Any:
    from langchain.llms.huggingface_text_gen_inference import (
        HuggingFaceTextGenInference,
    )

    return HuggingFaceTextGenInference


def _import_human() -> Any:
    from langchain.llms.human import HumanInputLLM

    return HumanInputLLM


def _import_javelin_ai_gateway() -> Any:
    from langchain.llms.javelin_ai_gateway import JavelinAIGateway

    return JavelinAIGateway


def _import_koboldai() -> Any:
    from langchain.llms.koboldai import KoboldApiLLM

    return KoboldApiLLM


def _import_llamacpp() -> Any:
    from langchain.llms.llamacpp import LlamaCpp

    return LlamaCpp


def _import_manifest() -> Any:
    from langchain.llms.manifest import ManifestWrapper

    return ManifestWrapper


def _import_minimax() -> Any:
    from langchain.llms.minimax import Minimax

    return Minimax


def _import_mlflow() -> Any:
    from langchain.llms.mlflow import Mlflow

    return Mlflow


def _import_mlflow_chat() -> Any:
    from langchain.chat_models.mlflow import ChatMlflow

    return ChatMlflow


def _import_mlflow_ai_gateway() -> Any:
    from langchain.llms.mlflow_ai_gateway import MlflowAIGateway

    return MlflowAIGateway


def _import_modal() -> Any:
    from langchain.llms.modal import Modal

    return Modal


def _import_mosaicml() -> Any:
    from langchain.llms.mosaicml import MosaicML

    return MosaicML


def _import_nlpcloud() -> Any:
    from langchain.llms.nlpcloud import NLPCloud

    return NLPCloud


def _import_octoai_endpoint() -> Any:
    from langchain.llms.octoai_endpoint import OctoAIEndpoint

    return OctoAIEndpoint


def _import_ollama() -> Any:
    from langchain.llms.ollama import Ollama

    return Ollama


def _import_opaqueprompts() -> Any:
    from langchain.llms.opaqueprompts import OpaquePrompts

    return OpaquePrompts


def _import_azure_openai() -> Any:
    from langchain.llms.openai import AzureOpenAI

    return AzureOpenAI


def _import_openai() -> Any:
    from langchain.llms.openai import OpenAI

    return OpenAI


def _import_openai_chat() -> Any:
    from langchain.llms.openai import OpenAIChat

    return OpenAIChat


def _import_openllm() -> Any:
    from langchain.llms.openllm import OpenLLM

    return OpenLLM


def _import_openlm() -> Any:
    from langchain.llms.openlm import OpenLM

    return OpenLM


def _import_pai_eas_endpoint() -> Any:
    from langchain.llms.pai_eas_endpoint import PaiEasEndpoint

    return PaiEasEndpoint


def _import_petals() -> Any:
    from langchain.llms.petals import Petals

    return Petals


def _import_pipelineai() -> Any:
    from langchain.llms.pipelineai import PipelineAI

    return PipelineAI


def _import_predibase() -> Any:
    from langchain.llms.predibase import Predibase

    return Predibase


def _import_predictionguard() -> Any:
    from langchain.llms.predictionguard import PredictionGuard

    return PredictionGuard


def _import_promptlayer() -> Any:
    from langchain.llms.promptlayer_openai import PromptLayerOpenAI

    return PromptLayerOpenAI


def _import_promptlayer_chat() -> Any:
    from langchain.llms.promptlayer_openai import PromptLayerOpenAIChat

    return PromptLayerOpenAIChat


def _import_replicate() -> Any:
    from langchain.llms.replicate import Replicate

    return Replicate


def _import_rwkv() -> Any:
    from langchain.llms.rwkv import RWKV

    return RWKV


def _import_sagemaker_endpoint() -> Any:
    from langchain.llms.sagemaker_endpoint import SagemakerEndpoint

    return SagemakerEndpoint


def _import_self_hosted() -> Any:
    from langchain.llms.self_hosted import SelfHostedPipeline

    return SelfHostedPipeline


def _import_self_hosted_hugging_face() -> Any:
    from langchain.llms.self_hosted_hugging_face import SelfHostedHuggingFaceLLM

    return SelfHostedHuggingFaceLLM


def _import_stochasticai() -> Any:
    from langchain.llms.stochasticai import StochasticAI

    return StochasticAI


def _import_symblai_nebula() -> Any:
    from langchain.llms.symblai_nebula import Nebula

    return Nebula


def _import_textgen() -> Any:
    from langchain.llms.textgen import TextGen

    return TextGen


def _import_titan_takeoff() -> Any:
    from langchain.llms.titan_takeoff import TitanTakeoff

    return TitanTakeoff


def _import_titan_takeoff_pro() -> Any:
    from langchain.llms.titan_takeoff_pro import TitanTakeoffPro

    return TitanTakeoffPro


def _import_together() -> Any:
    from langchain.llms.together import Together

    return Together


def _import_tongyi() -> Any:
    from langchain.llms.tongyi import Tongyi

    return Tongyi


def _import_vertex() -> Any:
    from langchain.llms.vertexai import VertexAI

    return VertexAI


def _import_vertex_model_garden() -> Any:
    from langchain.llms.vertexai import VertexAIModelGarden

    return VertexAIModelGarden


def _import_vllm() -> Any:
    from langchain.llms.vllm import VLLM

    return VLLM


def _import_vllm_openai() -> Any:
    from langchain.llms.vllm import VLLMOpenAI

    return VLLMOpenAI


def _import_writer() -> Any:
    from langchain.llms.writer import Writer

    return Writer


def _import_xinference() -> Any:
    from langchain.llms.xinference import Xinference

    return Xinference


def _import_yandex_gpt() -> Any:
    from langchain.llms.yandex import YandexGPT

    return YandexGPT


def _import_volcengine_maas() -> Any:
    from langchain.llms.volcengine_maas import VolcEngineMaasLLM

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
    elif name == "Arcee":
        return _import_arcee()
    elif name == "Aviary":
        return _import_aviary()
    elif name == "AzureMLOnlineEndpoint":
        return _import_azureml_endpoint()
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
        "writer": _import_writer,
        "xinference": _import_xinference,
        "javelin-ai-gateway": _import_javelin_ai_gateway,
        "qianfan_endpoint": _import_baidu_qianfan_endpoint,
        "yandex_gpt": _import_yandex_gpt,
        "VolcEngineMaasLLM": _import_volcengine_maas(),
    }
