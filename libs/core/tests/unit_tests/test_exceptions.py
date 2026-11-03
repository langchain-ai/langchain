from langchain_core.exceptions import LangChainException, ModelError, ModelRefusalError
from langchain_core.messages import AIMessage


def test_model_refusal_error() -> None:
    raw = AIMessage(content=[], response_metadata={"stop_reason": "refusal"})
    err = ModelRefusalError("Blocked.", raw_message=raw)

    assert isinstance(err, ModelError)
    assert isinstance(err, LangChainException)
    assert err.is_retryable is False
    assert err.raw_message is raw
    assert str(err) == "Blocked."


def test_model_refusal_error_defaults() -> None:
    err = ModelRefusalError()

    assert str(err) == "Model refused to produce the requested output"
    assert err.raw_message is None
