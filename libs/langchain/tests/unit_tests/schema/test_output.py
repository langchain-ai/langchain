from langchain.schema.output import GenerationChunk


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
    ), "GenerationChunk + GenerationChunk should be a GenerationChunk with merged generation_info"

    assert GenerationChunk(text="Hello, ") + GenerationChunk(
        text="world!", generation_info={"foo": "bar"}
    ) + GenerationChunk(text="!", generation_info={"baz": "foo"}) == GenerationChunk(
        text="Hello, world!!", generation_info={"foo": "bar", "baz": "foo"}
    ), "GenerationChunk + GenerationChunk should be a GenerationChunk with merged generation_info"
