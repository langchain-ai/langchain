from langchain_core.runnables.configurable import DynamicRunnable


def test_lc_namespace() -> None:
    assert DynamicRunnable.get_lc_namespace() == ["langchain", "schema", "runnable"]
