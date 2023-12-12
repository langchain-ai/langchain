from __future__ import annotations

from importlib.metadata import version
from typing import Optional, Union

import httpx
from packaging.version import parse


def is_openai_v1() -> bool:
    _version = parse(version("openai"))
    return _version.major >= 1


def configure_http_client_by_proxy(
    proxies: Optional[str] = None
) -> Union[httpx.Client, None]:
    if proxies is None or not is_openai_v1():
        return None
    return httpx.Client(proxies=proxies)


def configure_http_async_client_by_proxy(
    proxies: Optional[str] = None
) -> Union[httpx.AsyncClient, None]:
    if proxies is None or not is_openai_v1():
        return None
    return httpx.AsyncClient(proxies=proxies)