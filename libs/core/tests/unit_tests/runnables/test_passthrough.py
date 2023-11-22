from langchain_core.runnables import RunnablePassthrough


def test_lc_namespace() -> None:
    assert RunnablePassthrough.get_lc_namespace() == ["langchain", "schema", "runnable"]
