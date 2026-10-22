"""Select the HTTPX-compatible backend used by `langchain-core`.

The maintained `httpx2` fork is preferred when it is installed, with a fallback to
`httpx` so that environments that only have the original package keep working.
This mirrors the selection made by `langsmith`, so both packages share one HTTP
stack.

Every module in `langchain-core` must import HTTPX through here, so that client,
transport, request and exception types all come from a single backend.

Type checkers resolve against `httpx`, whose API `httpx2` mirrors.
"""

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx2 as httpx
    except ModuleNotFoundError:
        import httpx
    else:
        # `httpx2` defers importing its connection layer until the first
        # transport is built, which would block a running event loop the first
        # time an async client is created. Import it eagerly, as `httpx` does.
        # Best effort: some platforms (e.g. emscripten) ship `httpx2` without
        # `httpcore2`.
        with contextlib.suppress(ImportError):
            import httpcore2  # noqa: F401

__all__ = ["httpx"]
