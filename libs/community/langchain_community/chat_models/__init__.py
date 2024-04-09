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
from typing import Any

_module_lookup = {
    "AzureChatOpenAI": "langchain_community.chat_models.azure_openai",
    "BedrockChat": "langchain_community.chat_models.bedrock",
    "ChatAnthropic": "langchain_community.chat_models.anthropic",
    "ChatAnyscale": "langchain_community.chat_models.anyscale",
    "ChatBaichuan": "langchain_community.chat_models.baichuan",
    "ChatCohere": "langchain_community.chat_models.cohere",
    "ChatDatabricks": "langchain_community.chat_models.databricks",
    "ChatDeepInfra": "langchain_community.chat_models.deepinfra",
    "ChatEverlyAI": "langchain_community.chat_models.everlyai",
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
    "ChatOllama": "langchain_community.chat_models.ollama",
    "ChatOpenAI": "langchain_community.chat_models.openai",
    "ChatPerplexity": "langchain_community.chat_models.perplexity",
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
    "PaiEasChatEndpoint": "langchain_community.chat_models.pai_eas_endpoint",
    "PromptLayerChatOpenAI": "langchain_community.chat_models.promptlayer_openai",
    "QianfanChatEndpoint": "langchain_community.chat_models.baidu_qianfan_endpoint",
    "VolcEngineMaasChat": "langchain_community.chat_models.volcengine_maas",
    "ChatPremAI": "langchain_community.chat_models.premai",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
