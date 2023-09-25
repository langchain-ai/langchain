import uuid
from http import HTTPStatus
from typing import Any, Dict

import httpx

from ..exceptions import AuthenticationError, ResponseError
from ..models import AccessToken


def _get_kwargs(token: str, scope: str) -> Dict[str, Any]:
    return {
        "method": "POST",
        "url": "/oauth",
        "data": {"scope": scope},
        "headers": {
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Bearer {token}",
        },
    }


def _build_response(response: httpx.Response) -> AccessToken:
    if response.status_code == HTTPStatus.OK:
        return AccessToken(**response.json())
    elif response.status_code == HTTPStatus.UNAUTHORIZED:
        raise AuthenticationError(
            response.url, response.status_code, response.content, response.headers
        )
    else:
        raise ResponseError(
            response.url, response.status_code, response.content, response.headers
        )


def sync(client: httpx.Client, token: str, scope: str) -> AccessToken:
    kwargs = _get_kwargs(token, scope)
    response = client.request(**kwargs)
    return _build_response(response)


async def asyncio(client: httpx.AsyncClient, token: str, scope: str) -> AccessToken:
    kwargs = _get_kwargs(token, scope)
    response = await client.request(**kwargs)
    return _build_response(response)
