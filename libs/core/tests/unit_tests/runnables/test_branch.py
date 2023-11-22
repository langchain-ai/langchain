from langchain_core.runnables import RunnableBranch


def test_lc_namespace() -> None:
    assert RunnableBranch.get_lc_namespace() == ["langchain", "schema", "runnable"]
