from __future__ import annotations

from unittest.mock import patch

import httpx

from langchain_community.utils.openai import (
    configure_http_async_client_by_proxy,
    configure_http_client_by_proxy,
)


def test_configure_http_client_by_proxy() -> None:
    actual = configure_http_client_by_proxy(None)
    assert actual is None

    actual2 = configure_http_async_client_by_proxy(None)
    assert actual is None

    proxies = "https://foobar.com"

    with patch(
        "langchain_community.utils.openai.is_openai_v1", side_effect=lambda: False
    ):
        actual = configure_http_client_by_proxy(proxies)
        assert actual is None

    with patch(
        "langchain_community.utils.openai.is_openai_v1", side_effect=lambda: True
    ):
        actual = configure_http_client_by_proxy(proxies)
        assert isinstance(actual, httpx.Client)

    with patch(
        "langchain_community.utils.openai.is_openai_v1", side_effect=lambda: False
    ):
        actual2 = configure_http_async_client_by_proxy(proxies)
        assert actual2 is None

    with patch(
        "langchain_community.utils.openai.is_openai_v1", side_effect=lambda: True
    ):
        actual2 = configure_http_async_client_by_proxy(proxies)
        assert isinstance(actual2, httpx.AsyncClient)
