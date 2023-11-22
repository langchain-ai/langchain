from langchain_core.runnables import RouterRunnable


def test_lc_namespace() -> None:
    assert RouterRunnable.get_lc_namespace() == ["langchain", "schema", "runnable"]
