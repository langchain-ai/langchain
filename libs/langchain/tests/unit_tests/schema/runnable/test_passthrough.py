from langchain_classic.schema.runnable.passthrough import __all__

EXPECTED_ALL = ["RunnableAssign", "RunnablePassthrough", "aidentity", "identity"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
