from langchain.memory.simple import SimpleMemory


def test_simple_memory() -> None:
    """Test SimpleMemory."""
    memory = SimpleMemory(memories={"baz": "foo"})

    output = memory.load_memory_variables({})

    assert output == {"baz": "foo"}
    assert ["baz"] == memory.memory_variables
