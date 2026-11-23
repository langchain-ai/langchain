"""Resolve the `httpx` library matching the installed `openai` SDK.

`openai>=3` is backed by `httpx2` (an API-identical drop-in for `httpx`),
`openai<3` by `httpx`. Transport/config objects injected into the SDK client
are built from the same library, so this module re-exports whichever one the
installed SDK version defaults to.

Import `httpx` from here **only** for objects handed to the OpenAI SDK client
(e.g. a custom transport, `Limits`, `Proxy`, or `http_client=`). Standalone HTTP
code should import `httpx` directly.

Note: the SDK's `DefaultHttpxClient`/`DefaultAsyncHttpxClient` classes are *not*
re-exported here — they are subclassable in both SDK versions (backed by
`httpx2` on `openai>=3`), so subclass `openai.DefaultHttpxClient` directly. The
`DefaultHttpx2Client` names are factory functions, not classes, and cannot be
subclassed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import openai


def _sdk_uses_httpx2() -> bool:
    """Whether the installed `openai` SDK defaults to `httpx2` (openai>=3)."""
    # First integer run, so tag-derived strings like `v3.0.0` still parse;
    # unparseable falls back to classic httpx (always present).
    match = re.match(r"\D*(\d+)", str(getattr(openai, "__version__", "")))
    return match is not None and int(match.group(1)) >= 3


if TYPE_CHECKING:
    # Type-check against `httpx`, whose API `httpx2` mirrors. This keeps mypy
    # working in the lint environment (which installs `httpx`, not `httpx2`)
    # while the runtime selection below picks the correct module.
    import httpx
elif _sdk_uses_httpx2():
    import httpx2 as httpx  # type: ignore[import-not-found]
else:
    import httpx

__all__ = ["httpx"]
