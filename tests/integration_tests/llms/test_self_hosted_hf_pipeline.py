"""Test HuggingFace Pipeline wrapper."""

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms.self_hosted_hf_pipeline import SelfHostedHuggingFacePipeline


def test_selfhosted_huggingface_pipeline_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = SelfHostedHuggingFacePipeline.from_model_id(
        model_id="gpt2", task="text-generation", model_kwargs={"max_new_tokens": 10}
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_selfhosted_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text generation model."""
    llm = SelfHostedHuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small", task="text2text-generation"
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_init_with_pipeline() -> None:
    """Test initialization with a HF pipeline."""
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    llm = SelfHostedHuggingFacePipeline(pipeline=pipe)
    output = llm("Say foo:")
    assert isinstance(output, str)
