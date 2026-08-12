"""Resolve the `httpx` library matching the installed `openai` SDK.

`openai>=3` is backed by `httpx2` (Pydantic's `httpx` fork; the public API
mirrors `httpx` closely, but the classes are *distinct types* —
`isinstance(httpx.Client(), httpx2.Client)` is `False`), `openai<3` by `httpx`.

The SDK accepts a fully built `http_client=` from either library and normalizes
it internally. What is *not* normalized are the pieces we assemble ourselves:
`_client_utils` builds its clients by subclassing `openai.DefaultHttpxClient`,
whose base is `httpx2.Client` on `openai>=3`, so the `Limits`/`Proxy`/transport
objects passed to that constructor are consumed by `httpx2` directly. Mixing
libraries there constructs successfully and then fails on the first request with
a bare `AssertionError` that the SDK reports as `APIConnectionError`. This
module re-exports whichever library the installed SDK is built on so those
objects always match.

Import `httpx` from here **only** for objects handed to the OpenAI SDK client
(e.g. a custom transport, `Limits`, or `Proxy`). Standalone HTTP code should
import `httpx` directly.

Note: the SDK's `DefaultHttpxClient`/`DefaultAsyncHttpxClient` classes are *not*
re-exported here — they are subclassable in both SDK versions (backed by
`httpx2` on `openai>=3`), so subclass `openai.DefaultHttpxClient` directly. The
`DefaultHttpx2Client` names are factory functions, not classes, and cannot be
subclassed.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import openai

logger = logging.getLogger(__name__)

# Top-level module names of the two supported HTTP libraries, most recent first.
_HTTPX_MODULES = ("httpx2", "httpx")


def _sdk_uses_httpx2() -> bool:
    """Whether the installed `openai` SDK is built on `httpx2` (openai>=3).

    Read off the SDK itself rather than inferred from its version string:
    `openai.DefaultHttpxClient` subclasses the client class of whichever library
    the SDK uses, so its MRO *is* the answer. `_client_utils` subclasses that
    same attribute, which makes a disagreement between this result and the
    library our transport objects must match structurally impossible.

    Falls back to the version string only if the SDK stops exposing a
    recognizable base class, warning so the guess is visible in logs rather
    than surfacing later as an opaque `APIConnectionError`.
    """
    for base in getattr(getattr(openai, "DefaultHttpxClient", None), "__mro__", ()):
        root = base.__module__.split(".", 1)[0]
        if root in _HTTPX_MODULES:
            return root == "httpx2"

    major = _sdk_major_version()
    if major is None:
        logger.warning(
            "Could not determine which httpx library the installed openai SDK "
            "(version %r) is built on: it exposes no recognizable "
            "DefaultHttpxClient base class and its version is unparseable. "
            "Falling back to classic `httpx`. If requests fail with "
            "`APIConnectionError`, pass an explicit `http_client=`.",
            getattr(openai, "__version__", None),
        )
        return False
    logger.warning(
        "The installed openai SDK (version %r) exposes no recognizable "
        "DefaultHttpxClient base class; falling back to its major version (%d) "
        "to select an httpx library.",
        getattr(openai, "__version__", None),
        major,
    )
    return major >= 3


def _sdk_major_version() -> int | None:
    """Major version of the installed `openai` SDK, or `None` if unreadable.

    Tolerates tag-derived strings such as `v3.0.0`, which build tooling
    sometimes leaves unstripped, by reading the first integer run.
    """
    match = re.match(r"\D*(\d+)", str(getattr(openai, "__version__", "")))
    return int(match.group(1)) if match else None


if TYPE_CHECKING:
    # Type-check against `httpx`, whose API `httpx2` mirrors. The runtime
    # branch below is not statically resolvable, and `httpx2` is absent
    # whenever `openai<3` is installed, so annotations pin the library that is
    # always available. Cross-library type mismatches are the trade-off.
    import httpx
elif _sdk_uses_httpx2():
    import httpx2 as httpx  # type: ignore[import-not-found]
else:
    import httpx

__all__ = ["httpx"]
