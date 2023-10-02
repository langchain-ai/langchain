from http import HTTPStatus
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx

from ..exceptions import AuthenticationError, ResponseError
from ..models import Chat, ChatCompletionChunk


def _get_kwargs(chat: Chat, token: Optional[str]) -> Dict[str, Any]:
    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-store",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return {
        "method": "POST",
        "url": "/chat/completions",
        "json": {**chat.dict(exclude_none=True), **{"stream": True}},
        "headers": headers,
    }


def _parse_chunk(line: str) -> Optional[ChatCompletionChunk]:
    name, _, value = line.partition(": ")
    if name == "data":
        if value == "[DONE]":
            return None
        else:
            return ChatCompletionChunk.parse_raw(value)
    else:
        return None


def _check_content_type(response: httpx.Response) -> None:
    content_type, _, _ = response.headers["content-type"].partition(";")
    if content_type != "text/event-stream":
        raise httpx.TransportError(
            "Expected response Content-Type to be 'text/event-stream',"
            f" got {content_type!r}"
        )


def _check_response(response: httpx.Response) -> None:
    if response.status_code == HTTPStatus.OK:
        _check_content_type(response)
    elif response.status_code == HTTPStatus.UNAUTHORIZED:
        raise AuthenticationError(
            response.url, response.status_code, b"", response.headers
        )
    else:
        raise ResponseError(response.url, response.status_code, b"", response.headers)


def sync(
    client: httpx.Client, chat: Chat, token: Optional[str]
) -> Iterator[ChatCompletionChunk]:
    kwargs = _get_kwargs(chat, token)
    with client.stream(**kwargs) as response:
        _check_response(response)
        for line in response.iter_lines():
            if chunk := _parse_chunk(line):
                yield chunk


async def asyncio(
    client: httpx.AsyncClient, chat: Chat, token: Optional[str]
) -> AsyncIterator[ChatCompletionChunk]:
    kwargs = _get_kwargs(chat, token)
    async with client.stream(**kwargs) as response:
        _check_response(response)
        async for line in response.aiter_lines():
            if chunk := _parse_chunk(line):
                yield chunk
