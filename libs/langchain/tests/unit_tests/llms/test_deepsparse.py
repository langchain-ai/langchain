import pytest

from langchain.llms import DeepSparse

generation_config = {"max_new_tokens": 5}


@pytest.mark.requires("deepsparse")
def test_deepsparse_call() -> None:
    """Test valid call to DeepSparse."""
    llm = DeepSparse(
        model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
        generation_config=generation_config,
    )
    output = llm("def ")
    assert isinstance(output, str)
    assert len(output) > 1


@pytest.mark.requires("deepsparse")
def test_deepsparse_streaming() -> None:
    """Test valid call to DeepSparse with streaming."""
    llm = DeepSparse(
        model="hf:neuralmagic/mpt-7b-chat-pruned50-quant",
        generation_config=generation_config,
        streaming=True,
    )

    output = " "
    for chunk in llm.stream("Tell me a joke", stop=["'", "\n"]):
        output += chunk

    assert isinstance(output, str)
    assert len(output) > 1


@pytest.mark.requires("deepsparse")
@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_deepsparse_astream() -> None:
    llm = DeepSparse(
        model="hf:neuralmagic/mpt-7b-chat-pruned50-quant",
        generation_config=generation_config,
    )
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
@pytest.mark.requires("deepsparse")
async def test_deepsparse_abatch() -> None:
    llm = DeepSparse(
        model="hf:neuralmagic/mpt-7b-chat-pruned50-quant",
        generation_config=generation_config,
    )
    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


@pytest.mark.asyncio
@pytest.mark.requires("deepsparse")
async def test_deepsparse_abatch_tags() -> None:
    llm = DeepSparse(
        model="hf:neuralmagic/mpt-7b-chat-pruned50-quant",
        generation_config=generation_config,
    )
    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
@pytest.mark.requires("deepsparse")
async def test_deepsparse_ainvoke() -> None:
    llm = DeepSparse(
        model="hf:neuralmagic/mpt-7b-chat-pruned50-quant",
        generation_config=generation_config,
    )
    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)
