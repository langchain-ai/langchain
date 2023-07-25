from langchain.schema.messages import HumanMessageChunk
from langchain.schema.output import ChatGenerationChunk, GenerationChunk


def test_generation_chunk() -> None:
    assert GenerationChunk(text="Hello, ") + GenerationChunk(
        text="world!"
    ) == GenerationChunk(
        text="Hello, world!"
    ), "GenerationChunk + GenerationChunk should be a GenerationChunk"

    assert GenerationChunk(text="Hello, ") + GenerationChunk(
        text="world!", generation_info={"foo": "bar"}
    ) == GenerationChunk(
        text="Hello, world!", generation_info={"foo": "bar"}
    ), "GenerationChunk + GenerationChunk should be a GenerationChunk with merged generation_info"  # noqa: E501

    assert GenerationChunk(text="Hello, ") + GenerationChunk(
        text="world!", generation_info={"foo": "bar"}
    ) + GenerationChunk(text="!", generation_info={"baz": "foo"}) == GenerationChunk(
        text="Hello, world!!", generation_info={"foo": "bar", "baz": "foo"}
    ), "GenerationChunk + GenerationChunk should be a GenerationChunk with merged generation_info"  # noqa: E501


def test_chat_generation_chunk() -> None:
    assert ChatGenerationChunk(
        message=HumanMessageChunk(content="Hello, ")
    ) + ChatGenerationChunk(
        message=HumanMessageChunk(content="world!")
    ) == ChatGenerationChunk(
        message=HumanMessageChunk(content="Hello, world!")
    ), "ChatGenerationChunk + ChatGenerationChunk should be a ChatGenerationChunk"

    assert ChatGenerationChunk(
        message=HumanMessageChunk(content="Hello, ")
    ) + ChatGenerationChunk(
        message=HumanMessageChunk(content="world!"), generation_info={"foo": "bar"}
    ) == ChatGenerationChunk(
        message=HumanMessageChunk(content="Hello, world!"),
        generation_info={"foo": "bar"},
    ), "GenerationChunk + GenerationChunk should be a GenerationChunk with merged generation_info"  # noqa: E501

    assert ChatGenerationChunk(
        message=HumanMessageChunk(content="Hello, ")
    ) + ChatGenerationChunk(
        message=HumanMessageChunk(content="world!"), generation_info={"foo": "bar"}
    ) + ChatGenerationChunk(
        message=HumanMessageChunk(content="!"), generation_info={"baz": "foo"}
    ) == ChatGenerationChunk(
        message=HumanMessageChunk(content="Hello, world!!"),
        generation_info={"foo": "bar", "baz": "foo"},
    ), "GenerationChunk + GenerationChunk should be a GenerationChunk with merged generation_info"  # noqa: E501
