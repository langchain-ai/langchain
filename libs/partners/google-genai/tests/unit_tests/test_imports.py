from langchain_google_genai import __all__

EXPECTED_ALL = [
    "ChatGoogleGenerativeAI",
    "GoogleGenerativeAIEmbeddings",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
