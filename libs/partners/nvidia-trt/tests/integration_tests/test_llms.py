"""Test TritonTensorRTLLM llm."""
import pytest

from langchain_nvidia_trt.llms import TritonTensorRTLLM

_MODEL_NAME = "ensemble"


@pytest.mark.skip(reason="Need a working Triton server")
def test_stream() -> None:
    """Test streaming tokens from NVIDIA TRT."""
    llm = TritonTensorRTLLM(model_name=_MODEL_NAME)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.skip(reason="Need a working Triton server")
async def test_astream() -> None:
    """Test streaming tokens from NVIDIA TRT."""
    llm = TritonTensorRTLLM(model_name=_MODEL_NAME)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.skip(reason="Need a working Triton server")
async def test_abatch() -> None:
    """Test streaming tokens from TritonTensorRTLLM."""
    llm = TritonTensorRTLLM(model_name=_MODEL_NAME)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


@pytest.mark.skip(reason="Need a working Triton server")
async def test_abatch_tags() -> None:
    """Test batch tokens from TritonTensorRTLLM."""
    llm = TritonTensorRTLLM(model_name=_MODEL_NAME)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


@pytest.mark.skip(reason="Need a working Triton server")
def test_batch() -> None:
    """Test batch tokens from TritonTensorRTLLM."""
    llm = TritonTensorRTLLM(model_name=_MODEL_NAME)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


@pytest.mark.skip(reason="Need a working Triton server")
async def test_ainvoke() -> None:
    """Test invoke tokens from TritonTensorRTLLM."""
    llm = TritonTensorRTLLM(model_name=_MODEL_NAME)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


@pytest.mark.skip(reason="Need a working Triton server")
def test_invoke() -> None:
    """Test invoke tokens from TritonTensorRTLLM."""
    llm = TritonTensorRTLLM(model_name=_MODEL_NAME)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)
