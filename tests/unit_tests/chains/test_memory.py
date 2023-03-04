from langchain.chains.base import SimpleMemory
from tests.unit_tests.chains.test_base import FakeChain


def test_simple_memory() -> None:
    """Test simple memory."""
    memory = SimpleMemory(memories={"baz": "zab"})
    chain = FakeChain(memory=memory)
    output = chain.run("bar")

    assert output == "baz"
    assert memory.memories == {"baz": "zab"}
    assert memory.memory_variables == ["baz"]
    assert memory.load_memory_variables() == {"baz": "zab"}
