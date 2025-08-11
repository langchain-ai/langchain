from langchain_openai import __all__
from langchain_openai.v1 import __all__ as v1_all

EXPECTED_ALL = [
    "OpenAI",
    "ChatOpenAI",
    "OpenAIEmbeddings",
    "AzureOpenAI",
    "AzureChatOpenAI",
    "AzureOpenAIEmbeddings",
    "custom_tool",
]

EXPECTED_ALL_V1 = ["ChatOpenAI"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
    assert sorted(EXPECTED_ALL_V1) == sorted(v1_all)
