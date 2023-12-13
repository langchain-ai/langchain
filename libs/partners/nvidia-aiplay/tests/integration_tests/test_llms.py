"""Test NVAIPlayLLM llm."""
from langchain_nvidia_aiplay.llms import NVAIPlayLLM


def test_integration_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = NVAIPlayLLM()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_integration_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = NVAIPlayLLM()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_integration_abatch() -> None:
    """Test streaming tokens from NVAIPlayLLM."""
    llm = NVAIPlayLLM()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_integration_abatch_tags() -> None:
    """Test batch tokens from NVAIPlayLLM."""
    llm = NVAIPlayLLM()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


def test_integration_batch() -> None:
    """Test batch tokens from NVAIPlayLLM."""
    llm = NVAIPlayLLM()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_integration_ainvoke() -> None:
    """Test invoke tokens from NVAIPlayLLM."""
    llm = NVAIPlayLLM()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


def test_integration_invoke() -> None:
    """Test invoke tokens from NVAIPlayLLM."""
    llm = NVAIPlayLLM()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)
