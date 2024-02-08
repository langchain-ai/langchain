from langchain import chat_models
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
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


def test_all_imports() -> None:
    assert set(chat_models.__all__) == set(EXPECTED_ALL)
    assert_all_importable(chat_models)
