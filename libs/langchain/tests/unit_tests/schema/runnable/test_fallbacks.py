from langchain.schema.runnable.fallbacks import __all__

EXPECTED_ALL = ["RunnableWithFallbacks"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
