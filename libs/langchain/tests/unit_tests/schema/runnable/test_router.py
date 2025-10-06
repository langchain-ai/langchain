from langchain_classic.schema.runnable.router import __all__

EXPECTED_ALL = ["RouterInput", "RouterRunnable"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
