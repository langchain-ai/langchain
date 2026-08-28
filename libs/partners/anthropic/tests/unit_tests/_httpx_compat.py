"""Resolve the `httpx` library the installed `anthropic` SDK is built on.

Test-only. `langchain_anthropic` hands the SDK nothing but primitives — a
`base_url` string, a float `timeout`, a `proxy` URL string — and gets its client
class by subclassing `anthropic.DefaultHttpxClient`, which already points at the
right library. So no `httpx` type crosses that boundary in library code and
nothing there needs to know which library is in play.

These tests do cross it: they inject mock transports, and the SDK accepts a
built client only from the library it is itself built on (`httpx` on
`anthropic<1`, `httpx2` on `anthropic>=1`), rejecting the other with a
`TypeError`. `anthropic.DefaultHttpxClient` subclasses that library's client
class, so its MRO names the one to use.
"""

from __future__ import annotations

import importlib

import anthropic

_HTTPX_MODULES = ("httpx2", "httpx")


def _resolve_httpx_module() -> str:
    """Name of the HTTP library `anthropic.DefaultHttpxClient` derives from.

    Raises:
        RuntimeError: If the SDK exposes no recognizable base class. Guessing
            from the version string would be wrong as often as it was right;
            failing at import makes the mismatch obvious here rather than in a
            confusing downstream `TypeError`.
    """
    for base in anthropic.DefaultHttpxClient.__mro__:
        root = base.__module__.split(".", 1)[0]
        if root in _HTTPX_MODULES:
            return root

    msg = (
        f"Cannot tell which httpx library anthropic "
        f"{getattr(anthropic, '__version__', 'unknown')} is built on: "
        f"DefaultHttpxClient derives from none of {_HTTPX_MODULES}."
    )
    raise RuntimeError(msg)


httpx = importlib.import_module(_resolve_httpx_module())
