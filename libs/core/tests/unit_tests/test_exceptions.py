from langchain_core.exceptions import LangChainException, ModelError, ModelRefusalError
from langchain_core.messages import AIMessage


def test_model_refusal_error_fields() -> None:
    raw = AIMessage(content=[], response_metadata={"stop_reason": "refusal"})
    err = ModelRefusalError(
        provider="anthropic",
        stop_reason="refusal",
        category="cyber",
        explanation="Request touches on malware development.",
        raw_message=raw,
    )

    assert isinstance(err, ModelError)
    assert isinstance(err, LangChainException)
    assert err.is_retryable is False
    assert err.provider == "anthropic"
    assert err.stop_reason == "refusal"
    assert err.category == "cyber"
    assert err.explanation == "Request touches on malware development."
    assert err.raw_message is raw
    assert str(err) == (
        "Model refused to produce the requested output (provider='anthropic', "
        "stop_reason='refusal', category='cyber'): "
        "Request touches on malware development."
    )


def test_model_refusal_error_defaults() -> None:
    err = ModelRefusalError()

    assert str(err) == "Model refused to produce the requested output"
    assert err.provider is None
    assert err.stop_reason is None
    assert err.category is None
    assert err.explanation is None
    assert err.raw_message is None


def test_model_refusal_error_explicit_message() -> None:
    err = ModelRefusalError("I can't help with that.", provider="openai")

    assert str(err) == "I can't help with that."
    assert err.provider == "openai"
