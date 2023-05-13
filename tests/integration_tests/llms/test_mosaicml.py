"""Test MosaicML API wrapper."""
from langchain.llms.mosaicml import MosaicLLM


def test_mosaicml_llm_call() -> None:
    """Test valid call to MosaicML."""
    llm = MosaicLLM(model_kwargs={})
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_mosaicml_endpoint_change() -> None:
    """Test valid call to MosaicML."""
    llm = MosaicLLM(
        endpoint_url="https://models.hosted-on.mosaicml.hosting/gpt2-xl/v1/predict"
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_mosaicml_extra_kwargs() -> None:
    llm = MosaicLLM(model_kwargs={"max_new_tokens": 1})
    assert llm.model_kwargs == {"max_new_tokens": 1}

    output = llm("Say foo:")

    assert isinstance(output, str)

    # should only generate one new token
    assert len(output.split()) == 1


def test_instruct_prompt() -> None:
    """Test instruct prompt."""
    llm = MosaicLLM(inject_instruction_format=True, model_kwargs={"do_sample": False})
    prompt = "Repeat the word foo"
    prompt = llm._transform_prompt(prompt)
    assert prompt.endswith("### Response:\n")
    output = llm(prompt)
    assert isinstance(output, str)
