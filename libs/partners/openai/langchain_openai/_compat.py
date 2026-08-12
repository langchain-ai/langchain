"""Resolve the `httpx` library matching the installed `openai` SDK.

`openai>=3` is backed by `httpx2` (an API-identical drop-in for `httpx`),
`openai<3` by `httpx`. Transport/config objects injected into the SDK client
must come from the same library, so this module re-exports whichever one the
SDK uses.

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

from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    # Type-check against `httpx`, whose API `httpx2` mirrors. This keeps mypy
    # working in the lint environment (which installs `httpx`, not `httpx2`)
    # while the runtime selection below picks the correct module.
    import httpx
elif hasattr(openai, "DefaultHttpx2Client"):
    # openai>=3: SDK client is backed by `httpx2`.
    import httpx2 as httpx  # type: ignore[import-not-found]
else:
    # openai<3: SDK client is backed by `httpx`.
    import httpx

__all__ = ["httpx"]
