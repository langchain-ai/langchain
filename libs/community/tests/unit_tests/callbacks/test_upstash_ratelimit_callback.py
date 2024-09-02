import logging
from typing import Any
from unittest.mock import create_autospec

import pytest
from langchain_core.outputs import LLMResult

from langchain_community.callbacks import UpstashRatelimitError, UpstashRatelimitHandler

logger = logging.getLogger(__name__)

try:
    from upstash_ratelimit import Ratelimit, Response
except ImportError:
    Ratelimit, Response = None, None


# Fixtures
@pytest.fixture
def request_ratelimit() -> Ratelimit:
    ratelimit = create_autospec(Ratelimit)
    response = Response(allowed=True, limit=10, remaining=10, reset=10000)
    ratelimit.limit.return_value = response
    return ratelimit


@pytest.fixture
def token_ratelimit() -> Ratelimit:
    ratelimit = create_autospec(Ratelimit)
    response = Response(allowed=True, limit=1000, remaining=1000, reset=10000)
    ratelimit.limit.return_value = response
    ratelimit.get_remaining.return_value = 1000
    return ratelimit


@pytest.fixture
def handler_with_both_limits(
    request_ratelimit: Ratelimit, token_ratelimit: Ratelimit
) -> UpstashRatelimitHandler:
    return UpstashRatelimitHandler(
        identifier="user123",
        token_ratelimit=token_ratelimit,
        request_ratelimit=request_ratelimit,
        include_output_tokens=False,
    )


# Tests
@pytest.mark.requires("upstash_ratelimit")
def test_init_no_limits() -> None:
    with pytest.raises(ValueError):
        UpstashRatelimitHandler(identifier="user123")


@pytest.mark.requires("upstash_ratelimit")
def test_init_request_limit_only(request_ratelimit: Ratelimit) -> None:
    handler = UpstashRatelimitHandler(
        identifier="user123", request_ratelimit=request_ratelimit
    )
    assert handler.request_ratelimit is not None
    assert handler.token_ratelimit is None


@pytest.mark.requires("upstash_ratelimit")
def test_init_token_limit_only(token_ratelimit: Ratelimit) -> None:
    handler = UpstashRatelimitHandler(
        identifier="user123", token_ratelimit=token_ratelimit
    )
    assert handler.token_ratelimit is not None
    assert handler.request_ratelimit is None


@pytest.mark.requires("upstash_ratelimit")
def test_on_chain_start_request_limit(handler_with_both_limits: Any) -> None:
    handler_with_both_limits.on_chain_start(serialized={}, inputs={})
    handler_with_both_limits.request_ratelimit.limit.assert_called_once_with("user123")
    handler_with_both_limits.token_ratelimit.limit.assert_not_called()


@pytest.mark.requires("upstash_ratelimit")
def test_on_chain_start_request_limit_reached(request_ratelimit: Any) -> None:
    request_ratelimit.limit.return_value = Response(
        allowed=False, limit=10, remaining=0, reset=10000
    )
    handler = UpstashRatelimitHandler(
        identifier="user123", token_ratelimit=None, request_ratelimit=request_ratelimit
    )
    with pytest.raises(UpstashRatelimitError):
        handler.on_chain_start(serialized={}, inputs={})


@pytest.mark.requires("upstash_ratelimit")
def test_on_llm_start_token_limit_reached(token_ratelimit: Any) -> None:
    token_ratelimit.get_remaining.return_value = 0
    handler = UpstashRatelimitHandler(
        identifier="user123", token_ratelimit=token_ratelimit, request_ratelimit=None
    )
    with pytest.raises(UpstashRatelimitError):
        handler.on_llm_start(serialized={}, prompts=["test"])


@pytest.mark.requires("upstash_ratelimit")
def test_on_llm_start_token_limit_reached_negative(token_ratelimit: Any) -> None:
    token_ratelimit.get_remaining.return_value = -10
    handler = UpstashRatelimitHandler(
        identifier="user123", token_ratelimit=token_ratelimit, request_ratelimit=None
    )
    with pytest.raises(UpstashRatelimitError):
        handler.on_llm_start(serialized={}, prompts=["test"])


@pytest.mark.requires("upstash_ratelimit")
def test_on_llm_end_with_token_limit(handler_with_both_limits: Any) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            }
        },
    )
    handler_with_both_limits.on_llm_end(response)
    handler_with_both_limits.token_ratelimit.limit.assert_called_once_with("user123", 2)


@pytest.mark.requires("upstash_ratelimit")
def test_on_llm_end_with_token_limit_include_output_tokens(
    token_ratelimit: Any,
) -> None:
    handler = UpstashRatelimitHandler(
        identifier="user123",
        token_ratelimit=token_ratelimit,
        request_ratelimit=None,
        include_output_tokens=True,
    )
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            }
        },
    )
    handler.on_llm_end(response)
    token_ratelimit.limit.assert_called_once_with("user123", 5)


@pytest.mark.requires("upstash_ratelimit")
def test_on_llm_end_without_token_usage(handler_with_both_limits: Any) -> None:
    response = LLMResult(generations=[], llm_output={})
    with pytest.raises(ValueError):
        handler_with_both_limits.on_llm_end(response)


@pytest.mark.requires("upstash_ratelimit")
def test_reset_handler(handler_with_both_limits: Any) -> None:
    new_handler = handler_with_both_limits.reset(identifier="user456")
    assert new_handler.identifier == "user456"
    assert not new_handler._checked


@pytest.mark.requires("upstash_ratelimit")
def test_reset_handler_no_new_identifier(handler_with_both_limits: Any) -> None:
    new_handler = handler_with_both_limits.reset()
    assert new_handler.identifier == "user123"
    assert not new_handler._checked


@pytest.mark.requires("upstash_ratelimit")
def test_on_chain_start_called_once(handler_with_both_limits: Any) -> None:
    handler_with_both_limits.on_chain_start(serialized={}, inputs={})
    handler_with_both_limits.on_chain_start(serialized={}, inputs={})
    assert handler_with_both_limits.request_ratelimit.limit.call_count == 1


@pytest.mark.requires("upstash_ratelimit")
def test_on_chain_start_reset_checked(handler_with_both_limits: Any) -> None:
    handler_with_both_limits.on_chain_start(serialized={}, inputs={})
    new_handler = handler_with_both_limits.reset(identifier="user456")
    new_handler.on_chain_start(serialized={}, inputs={})

    # becomes two because the mock object is kept in reset
    assert new_handler.request_ratelimit.limit.call_count == 2


@pytest.mark.requires("upstash_ratelimit")
def test_on_llm_start_no_token_limit(request_ratelimit: Any) -> None:
    handler = UpstashRatelimitHandler(
        identifier="user123", token_ratelimit=None, request_ratelimit=request_ratelimit
    )
    handler.on_llm_start(serialized={}, prompts=["test"])
    assert request_ratelimit.limit.call_count == 0


@pytest.mark.requires("upstash_ratelimit")
def test_on_llm_start_token_limit(handler_with_both_limits: Any) -> None:
    handler_with_both_limits.on_llm_start(serialized={}, prompts=["test"])
    assert handler_with_both_limits.token_ratelimit.get_remaining.call_count == 1


@pytest.mark.requires("upstash_ratelimit")
def test_full_chain_with_both_limits(handler_with_both_limits: Any) -> None:
    handler_with_both_limits.on_chain_start(serialized={}, inputs={})
    handler_with_both_limits.on_chain_start(serialized={}, inputs={})

    assert handler_with_both_limits.request_ratelimit.limit.call_count == 1
    assert handler_with_both_limits.token_ratelimit.limit.call_count == 0
    assert handler_with_both_limits.token_ratelimit.get_remaining.call_count == 0

    handler_with_both_limits.on_llm_start(serialized={}, prompts=["test"])

    assert handler_with_both_limits.request_ratelimit.limit.call_count == 1
    assert handler_with_both_limits.token_ratelimit.limit.call_count == 0
    assert handler_with_both_limits.token_ratelimit.get_remaining.call_count == 1

    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            }
        },
    )
    handler_with_both_limits.on_llm_end(response)

    assert handler_with_both_limits.request_ratelimit.limit.call_count == 1
    assert handler_with_both_limits.token_ratelimit.limit.call_count == 1
    assert handler_with_both_limits.token_ratelimit.get_remaining.call_count == 1
