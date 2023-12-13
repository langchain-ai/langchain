from langchain_google import __all__

EXPECTED_ALL = [
    "ChatGoogleGenerativeAI",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
