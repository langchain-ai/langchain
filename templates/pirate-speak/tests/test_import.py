from langchain_core.runnables import Runnable


def test_import() -> None:
    from pirate_speak.chain import chain

    assert isinstance(chain, Runnable)
