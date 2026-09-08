"""Mocked WalletForge x402 tests — no network and no live spend."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[4]
WRAPPER_MODULES = (
    "examples.community_tools.walletforge_x402",
    "examples.community_tools.walletforge_x402.tools",
)


@pytest.fixture(autouse=True)
def _repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def _drop_wrapper_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in WRAPPER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)


def _install_fake_langchain_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    fetch_tool = SimpleNamespace(name="fetch_markdown")
    normalize_tool = SimpleNamespace(name="normalize_text")

    def fake_fetch(buyer: Any = None) -> Any:
        return fetch_tool

    def fake_normalize(buyer: Any = None) -> Any:
        return normalize_tool

    def fake_both(buyer: Any = None) -> list[Any]:
        return [fetch_tool, normalize_tool]

    langchain_tools = ModuleType("walletforge_x402.langchain_tools")
    langchain_tools.fetch_markdown_tool = fake_fetch  # type: ignore[attr-defined]
    langchain_tools.normalize_text_tool = fake_normalize  # type: ignore[attr-defined]
    langchain_tools.walletforge_tools = fake_both  # type: ignore[attr-defined]

    parent = ModuleType("walletforge_x402")
    parent.langchain_tools = langchain_tools  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "walletforge_x402", parent)
    monkeypatch.setitem(
        sys.modules, "walletforge_x402.langchain_tools", langchain_tools
    )
    _drop_wrapper_modules(monkeypatch)
    return SimpleNamespace(fetch=fetch_tool, normalize=normalize_tool)


def test_fetch_markdown_unpaid_fixture_is_x402_v2() -> None:
    body = _load_fixture("fetch_markdown_402.json")
    assert body["x402Version"] == 2
    assert "PAYMENT-SIGNATURE" in body["error"]
    assert body["resource"]["url"] == "https://api.walletforge.app/v1/fetch-markdown"
    accept = body["accepts"][0]
    assert accept["amount"] == "50000"
    assert accept["network"] == "eip155:8453"
    assert accept["asset"].lower() == "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
    assert accept["payTo"].lower() == "0xb5f5a86df5f78ed78920f74a1d7f26368f708e94"
    assert body["extensions"]["bazaar"]["info"]["serviceName"] == "WalletForge"


def test_normalize_unpaid_fixture_is_x402_v2() -> None:
    body = _load_fixture("normalize_402.json")
    assert body["x402Version"] == 2
    assert body["resource"]["url"] == "https://api.walletforge.app/v1/normalize"
    accept = body["accepts"][0]
    assert accept["amount"] == "10000"
    assert accept["network"] == "eip155:8453"
    assert "bazaar" in body["extensions"]


def test_tools_reexport_with_mocked_package(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_langchain_tools(monkeypatch)
    from examples.community_tools.walletforge_x402.tools import (
        fetch_markdown_tool,
        normalize_text_tool,
        walletforge_tools,
    )

    tools = walletforge_tools()
    assert [tool.name for tool in tools] == ["fetch_markdown", "normalize_text"]
    assert fetch_markdown_tool() is fake.fetch
    assert normalize_text_tool() is fake.normalize


def test_package_init_exports_mocked_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_langchain_tools(monkeypatch)
    from examples.community_tools.walletforge_x402 import walletforge_tools

    assert [tool.name for tool in walletforge_tools()] == [
        "fetch_markdown",
        "normalize_text",
    ]


def test_import_error_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "walletforge_x402" or name.startswith("walletforge_x402."):
            raise ImportError("No module named 'walletforge_x402'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "walletforge_x402", raising=False)
    monkeypatch.delitem(sys.modules, "walletforge_x402.langchain_tools", raising=False)
    _drop_wrapper_modules(monkeypatch)

    from examples.community_tools.walletforge_x402.tools import fetch_markdown_tool

    with pytest.raises(ImportError, match="walletforge-x402"):
        fetch_markdown_tool()


def test_example_unpaid_challenge_does_not_send_payment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> MagicMock:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers")
        response = MagicMock()
        response.status_code = 402
        response.json.return_value = _load_fixture("fetch_markdown_402.json")
        return response

    monkeypatch.setattr("httpx.post", fake_post)
    from examples.community_tools.walletforge_x402.langchain_example import (
        unpaid_fetch_markdown_challenge,
    )

    response = unpaid_fetch_markdown_challenge()
    assert response.status_code == 402
    assert captured["url"] == "https://api.walletforge.app/v1/fetch-markdown"
    assert captured["json"]["url"] == "https://example.com"
    headers = captured["headers"] or {}
    assert "PAYMENT-SIGNATURE" not in {str(key).upper() for key in headers}


def test_example_main_unpaid_path_skips_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BUYER_PRIVATE_KEY", raising=False)

    class FakeResponse:
        status_code = 402

        def json(self) -> dict[str, Any]:
            return _load_fixture("fetch_markdown_402.json")

    from examples.community_tools.walletforge_x402 import langchain_example

    monkeypatch.setattr(
        langchain_example,
        "unpaid_fetch_markdown_challenge",
        lambda: FakeResponse(),
    )
    assert langchain_example.main([]) == 0
