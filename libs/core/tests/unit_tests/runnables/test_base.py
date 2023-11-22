from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.base import (
    RunnableBindingBase,
    RunnableEachBase,
    RunnableParallel,
)


def test_lc_namespace() -> None:
    assert RunnableSequence.get_lc_namespace() == ["langchain", "schema", "runnable"]
    assert RunnableEachBase.get_lc_namespace() == [
        "langchain",
        "schema",
        "runnable",
    ]
    assert RunnableParallel.get_lc_namespace() == [
        "langchain",
        "schema",
        "runnable",
    ]
    assert RunnableBindingBase.get_lc_namespace() == [
        "langchain",
        "schema",
        "runnable",
    ]
