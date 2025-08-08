"""Shims for asyncio features that may be missing from older python versions."""

import sys

if sys.version_info[:2] < (3, 11):
    from async_timeout import timeout as asyncio_timeout
else:
    from asyncio import timeout as asyncio_timeout


__all__ = ["asyncio_timeout"]
