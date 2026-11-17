from typing import Any

import pytest

from langchain_core.utils._gateway_redirect import _gateway_request_hooks


class Request:
    def __init__(self, url: str) -> None:
        self.url = url


@pytest.mark.parametrize(
    ("base_url", "target", "allowed"),
    [
        ("https://gateway.test", "https://GATEWAY.test:443/path", True),
        ("http://gateway.test", "http://gateway.test:80/path", True),
        ("https://gateway.test", "https://gateway.test:0/path", False),
        ("https://gateway.test", "http://gateway.test/path", False),
        ("https://gateway.test", "https://gateway.test:444/path", False),
        ("https://gateway.test", "https://gateway.test.attacker.test/path", False),
    ],
)
@pytest.mark.parametrize("is_async", [False, True])
async def test_gateway_request_hooks_limit_origin(
    base_url: str, target: str, *, allowed: bool, is_async: bool
) -> None:
    hook: Any = _gateway_request_hooks(base_url, is_async=is_async)["request"][0]

    async def invoke() -> None:
        if is_async:
            await hook(Request(target))
        else:
            hook(Request(target))

    if allowed:
        await invoke()
    else:
        with pytest.raises(ValueError, match="Cross-origin redirects"):
            await invoke()


@pytest.mark.parametrize(
    "url", [None, "relative/path", "ftp://gateway.test/path", "https:///path"]
)
def test_gateway_request_hooks_reject_invalid_origin(url: str | None) -> None:
    with pytest.raises(ValueError, match="absolute HTTP URL"):
        _gateway_request_hooks(url, is_async=False)
