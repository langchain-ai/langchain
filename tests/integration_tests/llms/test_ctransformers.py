"""Test C Transformers wrapper."""

from langchain.llms import CTransformers
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_ctransformers_call() -> None:
    """Test valid call to C Transformers."""
    config = {"max_new_tokens": 5}
    callback_handler = FakeCallbackHandler()

    llm = CTransformers(
        model="marella/gpt-2-ggml",
        config=config,
        callbacks=[callback_handler],
    )

    output = llm("Say foo:")
    assert isinstance(output, str)
    assert len(output) > 1
    assert 0 < callback_handler.llm_streams <= config["max_new_tokens"]
