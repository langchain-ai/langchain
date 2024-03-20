import pytest

from langchain import chat_models

EXPECTED_DEPRECATED_IMPORTS = [
    "ChatOpenAI",
    "BedrockChat",
    "AzureChatOpenAI",
    "FakeListChatModel",
    "PromptLayerChatOpenAI",
    "ChatEverlyAI",
    "ChatAnthropic",
    "ChatCohere",
    "ChatDatabricks",
    "ChatGooglePalm",
    "ChatMlflow",
    "ChatMLflowAIGateway",
    "ChatOllama",
    "ChatVertexAI",
    "JinaChat",
    "HumanInputChatModel",
    "MiniMaxChat",
    "ChatAnyscale",
    "ChatLiteLLM",
    "ErnieBotChat",
    "ChatJavelinAIGateway",
    "ChatKonko",
    "PaiEasChatEndpoint",
    "QianfanChatEndpoint",
    "ChatFireworks",
    "ChatYandexGPT",
    "ChatBaichuan",
    "ChatHunyuan",
    "GigaChat",
    "VolcEngineMaasChat",
]


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(chat_models, import_)
            assert "langchain_community" in e
    with pytest.raises(AttributeError):
        getattr(chat_models, "foo")
