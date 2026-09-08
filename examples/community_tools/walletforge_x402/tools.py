"""Re-export WalletForge x402 LangChain tools from the standalone package.

Install the extra before calling these factories:

```bash
pip install "git+https://github.com/mig26-design/walletforge-x402.git#egg=walletforge-x402[langchain]"
```

Paid tool calls spend real Base USDC. Unit tests mock the buyer and never settle.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = (
    "WalletForge x402 tools require the standalone package. Install with: "
    'pip install "git+https://github.com/mig26-design/walletforge-x402.git'
    '#egg=walletforge-x402[langchain]"'
)


def _langchain_tools() -> Any:
    try:
        from walletforge_x402 import langchain_tools
    except ImportError as exc:  # pragma: no cover - covered via mocked ImportError
        raise ImportError(_INSTALL_HINT) from exc
    return langchain_tools


def fetch_markdown_tool(buyer: Any | None = None) -> Any:
    """Return the paid `fetch_markdown` StructuredTool (0.05 USDC on Base)."""
    return _langchain_tools().fetch_markdown_tool(buyer)


def normalize_text_tool(buyer: Any | None = None) -> Any:
    """Return the paid `normalize_text` StructuredTool (0.01 USDC on Base)."""
    return _langchain_tools().normalize_text_tool(buyer)


def walletforge_tools(buyer: Any | None = None) -> list[Any]:
    """Return both WalletForge x402 StructuredTools bound to one buyer."""
    return _langchain_tools().walletforge_tools(buyer)


__all__ = [
    "fetch_markdown_tool",
    "normalize_text_tool",
    "walletforge_tools",
]
