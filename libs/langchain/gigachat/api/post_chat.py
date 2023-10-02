from http import HTTPStatus
from typing import Any, Dict, Optional

import httpx

from ..exceptions import AuthenticationError, ResponseError
from ..models import Chat, ChatCompletion


def _get_kwargs(chat: Chat, token: Optional[str]) -> Dict[str, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return {
        "method": "POST",
        "url": "/chat/completions",
        "json": chat.dict(exclude_none=True, exclude={"stream"}),
        "headers": headers,
    }


def _build_response(response: httpx.Response) -> ChatCompletion:
    if response.status_code == HTTPStatus.OK:
        return ChatCompletion(**response.json())
    elif response.status_code == HTTPStatus.UNAUTHORIZED:
        raise AuthenticationError(
            response.url, response.status_code, response.content, response.headers
        )
    else:
        raise ResponseError(
            response.url, response.status_code, response.content, response.headers
        )


def sync(client: httpx.Client, chat: Chat, token: Optional[str]) -> ChatCompletion:
    kwargs = _get_kwargs(chat, token)
    response = client.request(**kwargs)
    return _build_response(response)


async def asyncio(
    client: httpx.AsyncClient, chat: Chat, token: Optional[str]
) -> ChatCompletion:
    kwargs = _get_kwargs(chat, token)
    response = await client.request(**kwargs)
    return _build_response(response)
