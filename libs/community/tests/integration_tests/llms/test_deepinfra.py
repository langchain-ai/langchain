"""Test DeepInfra API wrapper."""

from langchain_community.llms.deepinfra import DeepInfra


def test_deepinfra_call() -> None:
    """Test valid call to DeepInfra."""
    llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
    output = llm.invoke("What is 2 + 2?")
    assert isinstance(output, str)


async def test_deepinfra_acall() -> None:
    llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
    output = await llm.ainvoke("What is 2 + 2?")
    assert llm._llm_type == "deepinfra"
    assert isinstance(output, str)


def test_deepinfra_stream() -> None:
    llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
    num_chunks = 0
    for chunk in llm.stream("[INST] Hello [/INST] "):
        num_chunks += 1
    assert num_chunks > 0


async def test_deepinfra_astream() -> None:
    llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
    num_chunks = 0
    async for chunk in llm.astream("[INST] Hello [/INST] "):
        num_chunks += 1
    assert num_chunks > 0
