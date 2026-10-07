

def test_should_retry_exception_accepts_bare_class() -> None:
    """A bare exception class must use isinstance, not truthy-predicate (#41025)."""
    from langchain.agents.middleware._retry import should_retry_exception

    assert not should_retry_exception(KeyError("k"), ValueError)
    assert not should_retry_exception(KeyError("k"), (ValueError,))
    assert should_retry_exception(ValueError("v"), ValueError)
    assert should_retry_exception(ValueError("v"), (KeyError, ValueError))
    assert not should_retry_exception(ValueError("v"), KeyError)
    # Predicate callables still work.
    assert should_retry_exception(ValueError("v"), lambda e: True)
    assert not should_retry_exception(ValueError("v"), lambda e: False)


def test_tool_retry_middleware_does_not_retry_excluded_errors() -> None:
    """`retry_on=TimeoutError` must re-raise a KeyError after one attempt (#41025)."""
    import pytest as _pytest

    from langchain.agents.middleware import ToolRetryMiddleware

    calls = {"n": 0}

    class Req:
        tool = None
        tool_call = {"name": "t", "id": "1", "args": {}}

    def handler(_req):
        calls["n"] += 1
        raise KeyError("not a timeout")

    mw = ToolRetryMiddleware(max_retries=2, retry_on=TimeoutError, initial_delay=0, jitter=False)
    with _pytest.raises(KeyError, match="not a timeout"):
        mw.wrap_tool_call(Req(), handler)
    assert calls["n"] == 1
