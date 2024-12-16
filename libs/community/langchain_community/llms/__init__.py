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

from langchain_core._api.deprecation import warn_deprecated
from langchain_core.language_models.llms import BaseLLM


def _import_ai21() -> Type[BaseLLM]:
    from langchain_community.llms.ai21 import AI21

    return AI21


def _import_aleph_alpha() -> Type[BaseLLM]:
    from langchain_community.llms.aleph_alpha import AlephAlpha

    return AlephAlpha


def _import_amazon_api_gateway() -> Type[BaseLLM]:
    from langchain_community.llms.amazon_api_gateway import AmazonAPIGateway

    return AmazonAPIGateway


def _import_anthropic() -> Type[BaseLLM]:
    from langchain_community.llms.anthropic import Anthropic

    return Anthropic


def _import_anyscale() -> Type[BaseLLM]:
    from langchain_community.llms.anyscale import Anyscale

    return Anyscale


def _import_aphrodite() -> Type[BaseLLM]:
    from langchain_community.llms.aphrodite import Aphrodite

    return Aphrodite


def _import_arcee() -> Type[BaseLLM]:
    from langchain_community.llms.arcee import Arcee

    return Arcee


def _import_aviary() -> Type[BaseLLM]:
    from langchain_community.llms.aviary import Aviary

    return Aviary


def _import_azureml_endpoint() -> Type[BaseLLM]:
    from langchain_community.llms.azureml_endpoint import AzureMLOnlineEndpoint

    return AzureMLOnlineEndpoint


def _import_baichuan() -> Type[BaseLLM]:
    from langchain_community.llms.baichuan import BaichuanLLM

    return BaichuanLLM


def _import_baidu_qianfan_endpoint() -> Type[BaseLLM]:
    from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint

    return QianfanLLMEndpoint


def _import_bananadev() -> Type[BaseLLM]:
    from langchain_community.llms.bananadev import Banana

    return Banana


def _import_baseten() -> Type[BaseLLM]:
    from langchain_community.llms.baseten import Baseten

    return Baseten


def _import_beam() -> Type[BaseLLM]:
    from langchain_community.llms.beam import Beam

    return Beam


def _import_bedrock() -> Type[BaseLLM]:
    from langchain_community.llms.bedrock import Bedrock

    return Bedrock


def _import_bigdlllm() -> Type[BaseLLM]:
    from langchain_community.llms.bigdl_llm import BigdlLLM

    return BigdlLLM


def _import_bittensor() -> Type[BaseLLM]:
    from langchain_community.llms.bittensor import NIBittensorLLM

    return NIBittensorLLM


def _import_cerebriumai() -> Type[BaseLLM]:
    from langchain_community.llms.cerebriumai import CerebriumAI

    return CerebriumAI


def _import_chatglm() -> Type[BaseLLM]:
    from langchain_community.llms.chatglm import ChatGLM

    return ChatGLM


def _import_clarifai() -> Type[BaseLLM]:
    from langchain_community.llms.clarifai import Clarifai

    return Clarifai


def _import_cohere() -> Type[BaseLLM]:
    from langchain_community.llms.cohere import Cohere

    return Cohere


def _import_ctransformers() -> Type[BaseLLM]:
    from langchain_community.llms.ctransformers import CTransformers

    return CTransformers


def _import_ctranslate2() -> Type[BaseLLM]:
    from langchain_community.llms.ctranslate2 import CTranslate2

    return CTranslate2


def _import_databricks() -> Type[BaseLLM]:
    from langchain_community.llms.databricks import Databricks

    return Databricks


# deprecated / only for back compat - do not add to __all__
def _import_databricks_chat() -> Any:
    warn_deprecated(
        since="0.0.22",
        removal="1.0",
        alternative_import="langchain_community.chat_models.ChatDatabricks",
    )
    from langchain_community.chat_models.databricks import ChatDatabricks

    return ChatDatabricks


def _import_deepinfra() -> Type[BaseLLM]:
    from langchain_community.llms.deepinfra import DeepInfra

    return DeepInfra


def _import_deepsparse() -> Type[BaseLLM]:
    from langchain_community.llms.deepsparse import DeepSparse

    return DeepSparse


def _import_edenai() -> Type[BaseLLM]:
    from langchain_community.llms.edenai import EdenAI

    return EdenAI


def _import_fake() -> Type[BaseLLM]:
    from langchain_community.llms.fake import FakeListLLM

    return FakeListLLM


def _import_fireworks() -> Type[BaseLLM]:
    from langchain_community.llms.fireworks import Fireworks

    return Fireworks


def _import_forefrontai() -> Type[BaseLLM]:
    from langchain_community.llms.forefrontai import ForefrontAI

    return ForefrontAI


def _import_friendli() -> Type[BaseLLM]:
    from langchain_community.llms.friendli import Friendli

    return Friendli


def _import_gigachat() -> Type[BaseLLM]:
    from langchain_community.llms.gigachat import GigaChat

    return GigaChat


def _import_google_palm() -> Type[BaseLLM]:
    from langchain_community.llms.google_palm import GooglePalm

    return GooglePalm


def _import_gooseai() -> Type[BaseLLM]:
    from langchain_community.llms.gooseai import GooseAI

    return GooseAI


def _import_gpt4all() -> Type[BaseLLM]:
    from langchain_community.llms.gpt4all import GPT4All

    return GPT4All


def _import_gradient_ai() -> Type[BaseLLM]:
    from langchain_community.llms.gradient_ai import GradientLLM

    return GradientLLM


def _import_huggingface_endpoint() -> Type[BaseLLM]:
    from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

    return HuggingFaceEndpoint


def _import_huggingface_hub() -> Type[BaseLLM]:
    from langchain_community.llms.huggingface_hub import HuggingFaceHub

    return HuggingFaceHub


def _import_huggingface_pipeline() -> Type[BaseLLM]:
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

    return HuggingFacePipeline


def _import_huggingface_text_gen_inference() -> Type[BaseLLM]:
    from langchain_community.llms.huggingface_text_gen_inference import (
        HuggingFaceTextGenInference,
    )

    return HuggingFaceTextGenInference


def _import_human() -> Type[BaseLLM]:
    from langchain_community.llms.human import HumanInputLLM

    return HumanInputLLM


def _import_ipex_llm() -> Type[BaseLLM]:
    from langchain_community.llms.ipex_llm import IpexLLM

    return IpexLLM


def _import_javelin_ai_gateway() -> Type[BaseLLM]:
    from langchain_community.llms.javelin_ai_gateway import JavelinAIGateway

    return JavelinAIGateway


def _import_koboldai() -> Type[BaseLLM]:
    from langchain_community.llms.koboldai import KoboldApiLLM

    return KoboldApiLLM


def _import_konko() -> Type[BaseLLM]:
    from langchain_community.llms.konko import Konko

    return Konko


def _import_llamacpp() -> Type[BaseLLM]:
    from langchain_community.llms.llamacpp import LlamaCpp

    return LlamaCpp


def _import_llamafile() -> Type[BaseLLM]:
    from langchain_community.llms.llamafile import Llamafile

    return Llamafile


def _import_manifest() -> Type[BaseLLM]:
    from langchain_community.llms.manifest import ManifestWrapper

    return ManifestWrapper


def _import_minimax() -> Type[BaseLLM]:
    from langchain_community.llms.minimax import Minimax

    return Minimax


def _import_mlflow() -> Type[BaseLLM]:
    from langchain_community.llms.mlflow import Mlflow

    return Mlflow


# deprecated / only for back compat - do not add to __all__
def _import_mlflow_chat() -> Any:
    warn_deprecated(
        since="0.0.22",
        removal="1.0",
        alternative_import="langchain_community.chat_models.ChatMlflow",
    )
    from langchain_community.chat_models.mlflow import ChatMlflow

    return ChatMlflow


def _import_mlflow_ai_gateway() -> Type[BaseLLM]:
    from langchain_community.llms.mlflow_ai_gateway import MlflowAIGateway

    return MlflowAIGateway


def _import_mlx_pipeline() -> Type[BaseLLM]:
    from langchain_community.llms.mlx_pipeline import MLXPipeline

    return MLXPipeline


def _import_modal() -> Type[BaseLLM]:
    from langchain_community.llms.modal import Modal

    return Modal


def _import_mosaicml() -> Type[BaseLLM]:
    from langchain_community.llms.mosaicml import MosaicML

    return MosaicML


def _import_nlpcloud() -> Type[BaseLLM]:
    from langchain_community.llms.nlpcloud import NLPCloud

    return NLPCloud


def _import_oci_md_tgi() -> Type[BaseLLM]:
    from langchain_community.llms.oci_data_science_model_deployment_endpoint import (
        OCIModelDeploymentTGI,
    )

    return OCIModelDeploymentTGI


def _import_oci_md_vllm() -> Type[BaseLLM]:
    from langchain_community.llms.oci_data_science_model_deployment_endpoint import (
        OCIModelDeploymentVLLM,
    )

    return OCIModelDeploymentVLLM


def _import_oci_md() -> Type[BaseLLM]:
    from langchain_community.llms.oci_data_science_model_deployment_endpoint import (
        OCIModelDeploymentLLM,
    )

    return OCIModelDeploymentLLM


def _import_oci_gen_ai() -> Type[BaseLLM]:
    from langchain_community.llms.oci_generative_ai import OCIGenAI

    return OCIGenAI


def _import_octoai_endpoint() -> Type[BaseLLM]:
    from langchain_community.llms.octoai_endpoint import OctoAIEndpoint

    return OctoAIEndpoint


def _import_ollama() -> Type[BaseLLM]:
    from langchain_community.llms.ollama import Ollama

    return Ollama


def _import_opaqueprompts() -> Type[BaseLLM]:
    from langchain_community.llms.opaqueprompts import OpaquePrompts

    return OpaquePrompts


def _import_azure_openai() -> Type[BaseLLM]:
    from langchain_community.llms.openai import AzureOpenAI

    return AzureOpenAI


def _import_openai() -> Type[BaseLLM]:
    from langchain_community.llms.openai import OpenAI

    return OpenAI


def _import_openai_chat() -> Type[BaseLLM]:
    from langchain_community.llms.openai import OpenAIChat

    return OpenAIChat


def _import_openllm() -> Type[BaseLLM]:
    from langchain_community.llms.openllm import OpenLLM

    return OpenLLM


def _import_openlm() -> Type[BaseLLM]:
    from langchain_community.llms.openlm import OpenLM

    return OpenLM


def _import_outlines() -> Type[BaseLLM]:
    from langchain_community.llms.outlines import Outlines

    return Outlines


def _import_pai_eas_endpoint() -> Type[BaseLLM]:
    from langchain_community.llms.pai_eas_endpoint import PaiEasEndpoint

    return PaiEasEndpoint


def _import_petals() -> Type[BaseLLM]:
    from langchain_community.llms.petals import Petals

    return Petals


def _import_pipelineai() -> Type[BaseLLM]:
    from langchain_community.llms.pipelineai import PipelineAI

    return PipelineAI


def _import_predibase() -> Type[BaseLLM]:
    from langchain_community.llms.predibase import Predibase

    return Predibase


def _import_predictionguard() -> Type[BaseLLM]:
    from langchain_community.llms.predictionguard import PredictionGuard

    return PredictionGuard


def _import_promptlayer() -> Type[BaseLLM]:
    from langchain_community.llms.promptlayer_openai import PromptLayerOpenAI

    return PromptLayerOpenAI


def _import_promptlayer_chat() -> Type[BaseLLM]:
    from langchain_community.llms.promptlayer_openai import PromptLayerOpenAIChat

    return PromptLayerOpenAIChat


def _import_replicate() -> Type[BaseLLM]:
    from langchain_community.llms.replicate import Replicate

    return Replicate


def _import_rwkv() -> Type[BaseLLM]:
    from langchain_community.llms.rwkv import RWKV

    return RWKV


def _import_sagemaker_endpoint() -> Type[BaseLLM]:
    from langchain_community.llms.sagemaker_endpoint import SagemakerEndpoint

    return SagemakerEndpoint


def _import_sambanovacloud() -> Type[BaseLLM]:
    from langchain_community.llms.sambanova import SambaNovaCloud

    return SambaNovaCloud


def _import_sambastudio() -> Type[BaseLLM]:
    from langchain_community.llms.sambanova import SambaStudio

    return SambaStudio


def _import_self_hosted() -> Type[BaseLLM]:
    from langchain_community.llms.self_hosted import SelfHostedPipeline

    return SelfHostedPipeline


def _import_self_hosted_hugging_face() -> Type[BaseLLM]:
    from langchain_community.llms.self_hosted_hugging_face import (
        SelfHostedHuggingFaceLLM,
    )

    return SelfHostedHuggingFaceLLM


def _import_stochasticai() -> Type[BaseLLM]:
    from langchain_community.llms.stochasticai import StochasticAI

    return StochasticAI


def _import_symblai_nebula() -> Type[BaseLLM]:
    from langchain_community.llms.symblai_nebula import Nebula

    return Nebula


def _import_textgen() -> Type[BaseLLM]:
    from langchain_community.llms.textgen import TextGen

    return TextGen


def _import_titan_takeoff() -> Type[BaseLLM]:
    from langchain_community.llms.titan_takeoff import TitanTakeoff

    return TitanTakeoff


def _import_titan_takeoff_pro() -> Type[BaseLLM]:
    from langchain_community.llms.titan_takeoff import TitanTakeoff

    return TitanTakeoff


def _import_together() -> Type[BaseLLM]:
    from langchain_community.llms.together import Together

    return Together


def _import_tongyi() -> Type[BaseLLM]:
    from langchain_community.llms.tongyi import Tongyi

    return Tongyi


def _import_vertex() -> Type[BaseLLM]:
    from langchain_community.llms.vertexai import VertexAI

    return VertexAI


def _import_vertex_model_garden() -> Type[BaseLLM]:
    from langchain_community.llms.vertexai import VertexAIModelGarden

    return VertexAIModelGarden


def _import_vllm() -> Type[BaseLLM]:
    from langchain_community.llms.vllm import VLLM

    return VLLM


def _import_vllm_openai() -> Type[BaseLLM]:
    from langchain_community.llms.vllm import VLLMOpenAI

    return VLLMOpenAI


def _import_watsonxllm() -> Type[BaseLLM]:
    from langchain_community.llms.watsonxllm import WatsonxLLM

    return WatsonxLLM


def _import_weight_only_quantization() -> Any:
    from langchain_community.llms.weight_only_quantization import (
        WeightOnlyQuantPipeline,
    )

    return WeightOnlyQuantPipeline


def _import_writer() -> Type[BaseLLM]:
    from langchain_community.llms.writer import Writer

    return Writer


def _import_xinference() -> Type[BaseLLM]:
    from langchain_community.llms.xinference import Xinference

    return Xinference


def _import_yandex_gpt() -> Type[BaseLLM]:
    from langchain_community.llms.yandex import YandexGPT

    return YandexGPT


def _import_yuan2() -> Type[BaseLLM]:
    from langchain_community.llms.yuan2 import Yuan2

    return Yuan2


def _import_volcengine_maas() -> Type[BaseLLM]:
    from langchain_community.llms.volcengine_maas import VolcEngineMaasLLM

    return VolcEngineMaasLLM


def _import_sparkllm() -> Type[BaseLLM]:
    from langchain_community.llms.sparkllm import SparkLLM

    return SparkLLM


def _import_you() -> Type[BaseLLM]:
    from langchain_community.llms.you import You

    return You


def _import_yi() -> Type[BaseLLM]:
    from langchain_community.llms.yi import YiLLM

    return YiLLM


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
    elif name == "BaichuanLLM" or name == "Baichuan":
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
    elif name == "BigdlLLM":
        return _import_bigdlllm()
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
    elif name == "Friendli":
        return _import_friendli()
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
    elif name == "IpexLLM":
        return _import_ipex_llm()
    elif name == "JavelinAIGateway":
        return _import_javelin_ai_gateway()
    elif name == "KoboldApiLLM":
        return _import_koboldai()
    elif name == "Konko":
        return _import_konko()
    elif name == "LlamaCpp":
        return _import_llamacpp()
    elif name == "Llamafile":
        return _import_llamafile()
    elif name == "ManifestWrapper":
        return _import_manifest()
    elif name == "Minimax":
        return _import_minimax()
    elif name == "Mlflow":
        return _import_mlflow()
    elif name == "MlflowAIGateway":
        return _import_mlflow_ai_gateway()
    elif name == "MLXPipeline":
        return _import_mlx_pipeline()
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
    elif name == "OCIModelDeploymentLLM":
        return _import_oci_md()
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
    elif name == "Outlines":
        return _import_outlines()
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
    elif name == "SambaNovaCloud":
        return _import_sambanovacloud()
    elif name == "SambaStudio":
        return _import_sambastudio()
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
    elif name == "WeightOnlyQuantPipeline":
        return _import_weight_only_quantization()
    elif name == "Writer":
        return _import_writer()
    elif name == "Xinference":
        return _import_xinference()
    elif name == "YandexGPT":
        return _import_yandex_gpt()
    elif name == "Yuan2":
        return _import_yuan2()
    elif name == "VolcEngineMaasLLM":
        return _import_volcengine_maas()
    elif name == "SparkLLM":
        return _import_sparkllm()
    elif name == "YiLLM":
        return _import_yi()
    elif name == "You":
        return _import_you()
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
    "BaichuanLLM",
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
    "Friendli",
    "GPT4All",
    "GigaChat",
    "GooglePalm",
    "GooseAI",
    "GradientLLM",
    "HuggingFaceEndpoint",
    "HuggingFaceHub",
    "HuggingFacePipeline",
    "HuggingFaceTextGenInference",
    "HumanInputLLM",
    "IpexLLM",
    "JavelinAIGateway",
    "KoboldApiLLM",
    "Konko",
    "LlamaCpp",
    "Llamafile",
    "ManifestWrapper",
    "Minimax",
    "Mlflow",
    "MlflowAIGateway",
    "MLXPipeline",
    "Modal",
    "MosaicML",
    "NIBittensorLLM",
    "NLPCloud",
    "Nebula",
    "OCIGenAI",
    "OCIModelDeploymentTGI",
    "OCIModelDeploymentVLLM",
    "OCIModelDeploymentLLM",
    "OctoAIEndpoint",
    "Ollama",
    "OpaquePrompts",
    "OpenAI",
    "OpenAIChat",
    "OpenLLM",
    "OpenLM",
    "Outlines",
    "PaiEasEndpoint",
    "Petals",
    "PipelineAI",
    "Predibase",
    "PredictionGuard",
    "PromptLayerOpenAI",
    "PromptLayerOpenAIChat",
    "QianfanLLMEndpoint",
    "RWKV",
    "Replicate",
    "SagemakerEndpoint",
    "SambaNovaCloud",
    "SambaStudio",
    "SelfHostedHuggingFaceLLM",
    "SelfHostedPipeline",
    "SparkLLM",
    "StochasticAI",
    "TextGen",
    "TitanTakeoff",
    "TitanTakeoffPro",
    "Together",
    "Tongyi",
    "VLLM",
    "VLLMOpenAI",
    "VertexAI",
    "VertexAIModelGarden",
    "VolcEngineMaasLLM",
    "WatsonxLLM",
    "WeightOnlyQuantPipeline",
    "Writer",
    "Xinference",
    "YandexGPT",
    "Yuan2",
    "YiLLM",
    "You",
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
        "baichuan": _import_baichuan,
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
        "databricks-chat": _import_databricks_chat,  # deprecated / only for back compat
        "deepinfra": _import_deepinfra,
        "deepsparse": _import_deepsparse,
        "edenai": _import_edenai,
        "fake-list": _import_fake,
        "forefrontai": _import_forefrontai,
        "friendli": _import_friendli,
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
        "llamafile": _import_llamafile,
        "textgen": _import_textgen,
        "minimax": _import_minimax,
        "mlflow": _import_mlflow,
        "mlflow-chat": _import_mlflow_chat,  # deprecated / only for back compat
        "mlflow-ai-gateway": _import_mlflow_ai_gateway,
        "mlx_pipeline": _import_mlx_pipeline,
        "modal": _import_modal,
        "mosaic": _import_mosaicml,
        "nebula": _import_symblai_nebula,
        "nibittensor": _import_bittensor,
        "nlpcloud": _import_nlpcloud,
        "oci_model_deployment_tgi_endpoint": _import_oci_md_tgi,
        "oci_model_deployment_vllm_endpoint": _import_oci_md_vllm,
        "oci_model_deployment_endpoint": _import_oci_md,
        "oci_generative_ai": _import_oci_gen_ai,
        "octoai_endpoint": _import_octoai_endpoint,
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
        "sambanovacloud": _import_sambanovacloud,
        "sambastudio": _import_sambastudio,
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
        "outlines": _import_outlines,
        "vllm": _import_vllm,
        "vllm_openai": _import_vllm_openai,
        "watsonxllm": _import_watsonxllm,
        "weight_only_quantization": _import_weight_only_quantization,
        "writer": _import_writer,
        "xinference": _import_xinference,
        "javelin-ai-gateway": _import_javelin_ai_gateway,
        "qianfan_endpoint": _import_baidu_qianfan_endpoint,
        "yandex_gpt": _import_yandex_gpt,
        "yuan2": _import_yuan2,
        "VolcEngineMaasLLM": _import_volcengine_maas,
        "SparkLLM": _import_sparkllm,
        "yi": _import_yi,
        "you": _import_you,
    }
