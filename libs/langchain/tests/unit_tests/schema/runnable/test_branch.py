from langchain.schema.runnable.branch import __all__

EXPECTED_ALL = ["RunnableBranch"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
