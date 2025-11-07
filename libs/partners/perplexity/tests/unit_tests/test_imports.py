from langchain_perplexity import __all__

EXPECTED_ALL = ["ChatPerplexity"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
