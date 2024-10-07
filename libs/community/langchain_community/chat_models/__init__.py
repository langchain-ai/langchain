"""**Chat Models** are a variation on language models.

While Chat Models use language models under the hood, the interface they expose
is a bit different. Rather than expose a "text in, text out" API, they expose
an interface where "chat messages" are the inputs and outputs.

**Class hierarchy:**

.. code-block::

    BaseLanguageModel --> BaseChatModel --> <name>  # Examples: ChatOpenAI, ChatGooglePalm

**Main helpers:**

.. code-block::

    AIMessage, BaseMessage, HumanMessage
"""  # noqa: E501

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.chat_models.anthropic import (
        ChatAnthropic,
    )
    from langchain_community.chat_models.anyscale import (
        ChatAnyscale,
    )
    from langchain_community.chat_models.azure_openai import (
        AzureChatOpenAI,
    )
    from langchain_community.chat_models.baichuan import (
        ChatBaichuan,
    )
    from langchain_community.chat_models.baidu_qianfan_endpoint import (
        QianfanChatEndpoint,
    )
    from langchain_community.chat_models.bedrock import (
        BedrockChat,
    )
    from langchain_community.chat_models.cohere import (
        ChatCohere,
    )
    from langchain_community.chat_models.coze import (
        ChatCoze,
    )
    from langchain_community.chat_models.databricks import (
        ChatDatabricks,
    )
    from langchain_community.chat_models.deepinfra import (
        ChatDeepInfra,
    )
    from langchain_community.chat_models.edenai import ChatEdenAI
    from langchain_community.chat_models.ernie import (
        ErnieBotChat,
    )
    from langchain_community.chat_models.everlyai import (
        ChatEverlyAI,
    )
    from langchain_community.chat_models.fake import (
        FakeListChatModel,
    )
    from langchain_community.chat_models.fireworks import (
        ChatFireworks,
    )
    from langchain_community.chat_models.friendli import (
        ChatFriendli,
    )
    from langchain_community.chat_models.gigachat import (
        GigaChat,
    )
    from langchain_community.chat_models.google_palm import (
        ChatGooglePalm,
    )
    from langchain_community.chat_models.gpt_router import (
        GPTRouter,
    )
    from langchain_community.chat_models.huggingface import (
        ChatHuggingFace,
    )
    from langchain_community.chat_models.human import (
        HumanInputChatModel,
    )
    from langchain_community.chat_models.hunyuan import (
        ChatHunyuan,
    )
    from langchain_community.chat_models.javelin_ai_gateway import (
        ChatJavelinAIGateway,
    )
    from langchain_community.chat_models.jinachat import (
        JinaChat,
    )
    from langchain_community.chat_models.kinetica import (
        ChatKinetica,
    )
    from langchain_community.chat_models.konko import (
        ChatKonko,
    )
    from langchain_community.chat_models.litellm import (
        ChatLiteLLM,
    )
    from langchain_community.chat_models.litellm_router import (
        ChatLiteLLMRouter,
    )
    from langchain_community.chat_models.llama_edge import (
        LlamaEdgeChatService,
    )
    from langchain_community.chat_models.llamacpp import ChatLlamaCpp
    from langchain_community.chat_models.maritalk import (
        ChatMaritalk,
    )
    from langchain_community.chat_models.minimax import (
        MiniMaxChat,
    )
    from langchain_community.chat_models.mlflow import (
        ChatMlflow,
    )
    from langchain_community.chat_models.mlflow_ai_gateway import (
        ChatMLflowAIGateway,
    )
    from langchain_community.chat_models.mlx import (
        ChatMLX,
    )
    from langchain_community.chat_models.moonshot import (
        MoonshotChat,
    )
    from langchain_community.chat_models.oci_generative_ai import (
        ChatOCIGenAI,  # noqa: F401
    )
    from langchain_community.chat_models.octoai import ChatOctoAI
    from langchain_community.chat_models.ollama import (
        ChatOllama,
    )
    from langchain_community.chat_models.openai import (
        ChatOpenAI,
    )
    from langchain_community.chat_models.pai_eas_endpoint import (
        PaiEasChatEndpoint,
    )
    from langchain_community.chat_models.perplexity import (
        ChatPerplexity,
    )
    from langchain_community.chat_models.premai import (
        ChatPremAI,
    )
    from langchain_community.chat_models.promptlayer_openai import (
        PromptLayerChatOpenAI,
    )
    from langchain_community.chat_models.sambanova import (
        ChatSambaNovaCloud,
    )
    from langchain_community.chat_models.snowflake import (
        ChatSnowflakeCortex,
    )
    from langchain_community.chat_models.solar import (
        SolarChat,
    )
    from langchain_community.chat_models.sparkllm import (
        ChatSparkLLM,
    )
    from langchain_community.chat_models.symblai_nebula import ChatNebula
    from langchain_community.chat_models.tongyi import (
        ChatTongyi,
    )
    from langchain_community.chat_models.vertexai import (
        ChatVertexAI,
    )
    from langchain_community.chat_models.volcengine_maas import (
        VolcEngineMaasChat,
    )
    from langchain_community.chat_models.yandex import (
        ChatYandexGPT,
    )
    from langchain_community.chat_models.yi import (
        ChatYi,
    )
    from langchain_community.chat_models.yuan2 import (
        ChatYuan2,
    )
    from langchain_community.chat_models.zhipuai import (
        ChatZhipuAI,
    )
__all__ = [
    "AzureChatOpenAI",
    "BedrockChat",
    "ChatAnthropic",
    "ChatAnyscale",
    "ChatBaichuan",
    "ChatCohere",
    "ChatCoze",
    "ChatOctoAI",
    "ChatDatabricks",
    "ChatDeepInfra",
    "ChatEdenAI",
    "ChatEverlyAI",
    "ChatFireworks",
    "ChatFriendli",
    "ChatGooglePalm",
    "ChatHuggingFace",
    "ChatHunyuan",
    "ChatJavelinAIGateway",
    "ChatKinetica",
    "ChatKonko",
    "ChatLiteLLM",
    "ChatLiteLLMRouter",
    "ChatMLX",
    "ChatMLflowAIGateway",
    "ChatMaritalk",
    "ChatMlflow",
    "ChatNebula",
    "ChatOCIGenAI",
    "ChatOllama",
    "ChatOpenAI",
    "ChatPerplexity",
    "ChatPremAI",
    "ChatSambaNovaCloud",
    "ChatSparkLLM",
    "ChatSnowflakeCortex",
    "ChatTongyi",
    "ChatVertexAI",
    "ChatYandexGPT",
    "ChatYuan2",
    "ChatZhipuAI",
    "ChatLlamaCpp",
    "ErnieBotChat",
    "FakeListChatModel",
    "GPTRouter",
    "GigaChat",
    "HumanInputChatModel",
    "JinaChat",
    "LlamaEdgeChatService",
    "MiniMaxChat",
    "MoonshotChat",
    "PaiEasChatEndpoint",
    "PromptLayerChatOpenAI",
    "QianfanChatEndpoint",
    "SolarChat",
    "VolcEngineMaasChat",
    "ChatYi",
]


_module_lookup = {
    "AzureChatOpenAI": "langchain_community.chat_models.azure_openai",
    "BedrockChat": "langchain_community.chat_models.bedrock",
    "ChatAnthropic": "langchain_community.chat_models.anthropic",
    "ChatAnyscale": "langchain_community.chat_models.anyscale",
    "ChatBaichuan": "langchain_community.chat_models.baichuan",
    "ChatCohere": "langchain_community.chat_models.cohere",
    "ChatCoze": "langchain_community.chat_models.coze",
    "ChatDatabricks": "langchain_community.chat_models.databricks",
    "ChatDeepInfra": "langchain_community.chat_models.deepinfra",
    "ChatEverlyAI": "langchain_community.chat_models.everlyai",
    "ChatEdenAI": "langchain_community.chat_models.edenai",
    "ChatFireworks": "langchain_community.chat_models.fireworks",
    "ChatFriendli": "langchain_community.chat_models.friendli",
    "ChatGooglePalm": "langchain_community.chat_models.google_palm",
    "ChatHuggingFace": "langchain_community.chat_models.huggingface",
    "ChatHunyuan": "langchain_community.chat_models.hunyuan",
    "ChatJavelinAIGateway": "langchain_community.chat_models.javelin_ai_gateway",
    "ChatKinetica": "langchain_community.chat_models.kinetica",
    "ChatKonko": "langchain_community.chat_models.konko",
    "ChatLiteLLM": "langchain_community.chat_models.litellm",
    "ChatLiteLLMRouter": "langchain_community.chat_models.litellm_router",
    "ChatMLflowAIGateway": "langchain_community.chat_models.mlflow_ai_gateway",
    "ChatMLX": "langchain_community.chat_models.mlx",
    "ChatMaritalk": "langchain_community.chat_models.maritalk",
    "ChatMlflow": "langchain_community.chat_models.mlflow",
    "ChatNebula": "langchain_community.chat_models.symblai_nebula",
    "ChatOctoAI": "langchain_community.chat_models.octoai",
    "ChatOCIGenAI": "langchain_community.chat_models.oci_generative_ai",
    "ChatOllama": "langchain_community.chat_models.ollama",
    "ChatOpenAI": "langchain_community.chat_models.openai",
    "ChatPerplexity": "langchain_community.chat_models.perplexity",
    "ChatSambaNovaCloud": "langchain_community.chat_models.sambanova",
    "ChatSnowflakeCortex": "langchain_community.chat_models.snowflake",
    "ChatSparkLLM": "langchain_community.chat_models.sparkllm",
    "ChatTongyi": "langchain_community.chat_models.tongyi",
    "ChatVertexAI": "langchain_community.chat_models.vertexai",
    "ChatYandexGPT": "langchain_community.chat_models.yandex",
    "ChatYuan2": "langchain_community.chat_models.yuan2",
    "ChatZhipuAI": "langchain_community.chat_models.zhipuai",
    "ErnieBotChat": "langchain_community.chat_models.ernie",
    "FakeListChatModel": "langchain_community.chat_models.fake",
    "GPTRouter": "langchain_community.chat_models.gpt_router",
    "GigaChat": "langchain_community.chat_models.gigachat",
    "HumanInputChatModel": "langchain_community.chat_models.human",
    "JinaChat": "langchain_community.chat_models.jinachat",
    "LlamaEdgeChatService": "langchain_community.chat_models.llama_edge",
    "MiniMaxChat": "langchain_community.chat_models.minimax",
    "MoonshotChat": "langchain_community.chat_models.moonshot",
    "PaiEasChatEndpoint": "langchain_community.chat_models.pai_eas_endpoint",
    "PromptLayerChatOpenAI": "langchain_community.chat_models.promptlayer_openai",
    "SolarChat": "langchain_community.chat_models.solar",
    "QianfanChatEndpoint": "langchain_community.chat_models.baidu_qianfan_endpoint",
    "VolcEngineMaasChat": "langchain_community.chat_models.volcengine_maas",
    "ChatPremAI": "langchain_community.chat_models.premai",
    "ChatLlamaCpp": "langchain_community.chat_models.llamacpp",
    "ChatYi": "langchain_community.chat_models.yi",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
