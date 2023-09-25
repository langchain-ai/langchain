from http import HTTPStatus
from typing import Any, Dict

import httpx

from ..exceptions import AuthenticationError, ResponseError
from ..models import Token


def _get_kwargs(user: str, password: str) -> Dict[str, Any]:
    return {
        "method": "POST",
        "url": "/token",
        "auth": (user, password),
    }


def _build_response(response: httpx.Response) -> Token:
    if response.status_code == HTTPStatus.OK:
        return Token(**response.json())
    elif response.status_code == HTTPStatus.UNAUTHORIZED:
        raise AuthenticationError(
            response.url, response.status_code, response.content, response.headers
        )
    else:
        raise ResponseError(
            response.url, response.status_code, response.content, response.headers
        )


def sync(client: httpx.Client, user: str, password: str) -> Token:
    kwargs = _get_kwargs(user, password)
    response = client.request(**kwargs)
    return _build_response(response)


async def asyncio(client: httpx.AsyncClient, user: str, password: str) -> Token:
    kwargs = _get_kwargs(user, password)
    response = await client.request(**kwargs)
    return _build_response(response)
