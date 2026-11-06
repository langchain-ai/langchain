"""`include_raw` must survive truncated Responses structured output.

`responses.with_raw_response.parse()` validates the Pydantic schema inside the
SDK call. A `max_output_tokens` truncation leaves mid-JSON text, and that
`ValidationError` used to escape before `with_structured_output` could return
the raw envelope (`status`, `incomplete_details`).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response import IncompleteDetails, Response
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_openai._compat import httpx

_TRUNCATED_JSON = '{"answer": "hel'


class _Answer(BaseModel):
    answer: str


def _responses_body(
    *,
    text: str,
    status: str,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    response = Response(
        id="resp_structured",
        created_at=1_700_000_000,
        model="gpt-4.1-mini",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="none",
        status=status,  # type: ignore[arg-type]
        incomplete_details=(
            IncompleteDetails(reason=incomplete_reason)  # type: ignore[arg-type]
            if incomplete_reason
            else None
        ),
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_structured",
                status=status,  # type: ignore[arg-type]
                role="assistant",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text=text,
                        annotations=[],
                    )
                ],
            )
        ],
    )
    body = response.model_dump(mode="json")
    # Real Responses payloads echo the structured format. Keep it so recovery
    # still has to tolerate `text.format == json_schema` plus invalid JSON.
    body["text"] = {
        "format": {
            "type": "json_schema",
            "name": "Answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        }
    }
    return body


def _chat_model(
    body: dict[str, Any],
) -> tuple[ChatOpenAI, httpx.Client, httpx.AsyncClient]:
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    async def arespond(request: httpx.Request) -> httpx.Response:
        return respond(request)

    sync_client = httpx.Client(transport=httpx.MockTransport(respond))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(arespond))
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key="test-key",  # type: ignore[arg-type]
        use_responses_api=True,
        max_retries=0,
        base_url="https://api.openai.com/v1",
        http_client=sync_client,
        http_async_client=async_client,
    )
    llm._seen_payloads = seen  # type: ignore[attr-defined]
    return llm, sync_client, async_client


def _output_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
    return "".join(parts)


@pytest.mark.parametrize("invoke_async", [False, True])
async def test_include_raw_returns_incomplete_responses_metadata(
    invoke_async: bool,
) -> None:
    """Truncated Responses JSON must return raw metadata instead of raising."""
    body = _responses_body(
        text=_TRUNCATED_JSON,
        status="incomplete",
        incomplete_reason="max_output_tokens",
    )
    llm, sync_client, async_client = _chat_model(body)
    structured = llm.with_structured_output(
        _Answer, method="json_schema", include_raw=True
    )
    try:
        result = (
            await structured.ainvoke("Say hello")
            if invoke_async
            else structured.invoke("Say hello")
        )
    finally:
        sync_client.close()
        await async_client.aclose()

    assert isinstance(result, dict)
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], ValidationError)
    raw = result["raw"]
    assert isinstance(raw, AIMessage)
    assert raw.response_metadata["status"] == "incomplete"
    assert raw.response_metadata["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
    assert _output_text(raw) == _TRUNCATED_JSON
    assert llm._seen_payloads  # type: ignore[attr-defined]
    assert "_defer_responses_parse_errors" not in llm._seen_payloads[0]  # type: ignore[attr-defined]


def test_include_raw_false_still_raises_on_truncated_responses_output() -> None:
    """Without `include_raw`, truncated structured output still fails closed."""
    body = _responses_body(
        text=_TRUNCATED_JSON,
        status="incomplete",
        incomplete_reason="max_output_tokens",
    )
    llm, sync_client, _async_client = _chat_model(body)
    structured = llm.with_structured_output(_Answer, method="json_schema")
    try:
        with pytest.raises(ValidationError):
            structured.invoke("Say hello")
    finally:
        sync_client.close()


def test_include_raw_keeps_strict_schema_errors_on_completed_responses() -> None:
    """Well-formed JSON that misses the schema still raises under `include_raw`."""
    body = _responses_body(text='{"answer": 1}', status="completed")
    llm, sync_client, _async_client = _chat_model(body)
    structured = llm.with_structured_output(
        _Answer, method="json_schema", include_raw=True
    )
    try:
        with pytest.raises(ValidationError):
            structured.invoke("Say hello")
    finally:
        sync_client.close()


@pytest.mark.parametrize("include_raw", [False, True])
def test_responses_structured_output_parses_completed_json(
    *, include_raw: bool
) -> None:
    """A completed payload that matches the schema still parses."""
    body = _responses_body(text='{"answer": "hello"}', status="completed")
    llm, sync_client, _async_client = _chat_model(body)
    structured = llm.with_structured_output(
        _Answer, method="json_schema", include_raw=include_raw
    )
    try:
        result = structured.invoke("Say hello")
    finally:
        sync_client.close()

    if include_raw:
        assert isinstance(result, dict)
        assert result["parsing_error"] is None
        assert result["parsed"] == _Answer(answer="hello")
        assert isinstance(result["raw"], AIMessage)
    else:
        assert result == _Answer(answer="hello")
