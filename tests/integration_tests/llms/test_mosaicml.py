"""Test MosaicML API wrapper."""
from langchain.llms.mosaicml import PROMPT_FOR_GENERATION_FORMAT, MosaicML


def test_mosaicml_llm_call() -> None:
    """Test valid call to MosaicML."""
    llm = MosaicML(model_kwargs={})
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_mosaicml_endpoint_change() -> None:
    """Test valid call to MosaicML."""
    llm = MosaicML(
        endpoint_url="https://models.hosted-on.mosaicml.hosting/gpt2-xl/v1/predict"
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_mosaicml_extra_kwargs() -> None:
    llm = MosaicML(model_kwargs={"max_new_tokens": 1})
    assert llm.model_kwargs == {"max_new_tokens": 1}

    output = llm("Say foo:")

    assert isinstance(output, str)

    # should only generate one new token
    assert len(output.split()) == 1


def test_instruct_prompt() -> None:
    """Test instruct prompt."""
    llm = MosaicML(inject_instruction_format=True, model_kwargs={"do_sample": False})
    instruction = "Repeat the word foo"
    prompt = llm._transform_prompt(instruction)
    expected_prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
    assert prompt == expected_prompt
    output = llm(prompt)
    assert isinstance(output, str)
