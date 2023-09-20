"""Test GradientAI API wrapper.

In order to run this test, you need to have an GradientAI api key.
You can get it by registering for free at https://gradient.ai/.

You'll then need to set:
- `GRADIENT_ACCESS_TOKEN` environment variable to your api key.
- `GRADIENT_WORKSPACE_ID` environment variable to your workspace id.
- `GRADIENT_MODEL_ID` environment variable to your workspace id.
"""
import os

from langchain.llms import GradientLLM


def test_gradient_acall() -> None:
    """Test simple call to gradient.ai."""
    model_id = os.environ["GRADIENT_MODEL_ID"]
    llm = GradientLLM(model_id=model_id)
    output = llm("Say hello:", temperature=0.2, max_tokens=250)

    assert llm._llm_type == "gradient"

    assert isinstance(output, str)
    assert len(output)


async def test_gradientai_acall() -> None:
    """Test async call to gradient.ai."""
    model_id = os.environ["GRADIENT_MODEL_ID"]
    llm = GradientLLM(model_id=model_id)
    output = await llm.agenerate(["Say hello:"], temperature=0.2, max_tokens=250)
    assert llm._llm_type == "gradient"

    assert isinstance(output, str)
    assert len(output)
