from langchain import llms


def test_all_imports() -> None:
    """Simple test to make sure all things can be imported."""
    for cls in llms.__all__:
        exec(f"from langchain.llms import {cls}")
        # TODO: assert all is of type BaseLLM
