from langchain import llms
from langchain.llms.base import BaseLLM


def test_all_imports() -> None:
    """Simple test to make sure all things can be imported."""
    for cls in llms.__all__:
        assert issubclass(getattr(llms, cls), BaseLLM)
