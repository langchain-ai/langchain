from langchain_core.runnables import RunnableWithFallbacks


def test_lc_namespace() -> None:
    assert RunnableWithFallbacks.get_lc_namespace() == [
        "langchain",
        "schema",
        "runnable",
    ]
