"""Test Beam API wrapper."""

from langchain.llms.beam import Beam


def test_beam_call() -> None:
    """Test valid call to Beam."""
    llm = Beam(
        model_name="gpt2",
        name="langchain-gpt2",
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        python_version="python3.8",
        python_packages=[
            "diffusers[torch]>=0.10",
            "transformers",
            "torch",
            "pillow",
            "accelerate",
            "safetensors",
            "xformers",
        ],
        max_length="5",
    )
    llm._deploy()

    output = llm._call("Your prompt goes here")
    assert isinstance(output, str)
