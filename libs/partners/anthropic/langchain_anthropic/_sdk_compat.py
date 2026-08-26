"""Adapt to the installed `anthropic` SDK major version.

`langchain-anthropic` supports `anthropic` 0.x and 1.x. Two differences between
them reach this package, and each is resolved here by reading the answer off the
installed SDK rather than off its version string:

- **Sampling parameters.** `anthropic>=1` dropped `temperature`, `top_p` and
  `top_k` from its message-creation methods, since current models do not use
  them. The API still honors them for older models, so they are relocated to
  `extra_body`, which the SDK merges into the request JSON as-is. The wire
  payload is identical on both majors.

- **Raw responses.** `.with_raw_response` on the async client returns
  `AsyncAPIResponse`, whose `parse()` is a coroutine; `anthropic<1` returned
  `LegacyAPIResponse` with a synchronous `parse()`.

`anthropic>=1` also moved its HTTP layer from `httpx` to
[`httpx2`](https://github.com/pydantic/httpx2), but that does not surface here:
this package hands the SDK only primitives (a `base_url` string, a float
`timeout`, a `proxy` URL string) and subclasses `anthropic.DefaultHttpxClient`,
which already points at the right library, so no `httpx` type crosses the
boundary. Only the unit tests inject `httpx` objects, and they resolve the
matching library themselves in `tests/unit_tests/_httpx_compat.py`.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# Sampling parameters `anthropic>=1` no longer accepts as named arguments.
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def _sdk_major_version() -> int | None:
    """Major version of the installed `anthropic` SDK, or `None` if unreadable.

    Tolerates tag-derived strings such as `v1.0.0`, which build tooling
    sometimes leaves unstripped, by reading the first integer run.
    """
    match = re.match(r"\D*(\d+)", str(getattr(anthropic, "__version__", "")))
    return int(match.group(1)) if match else None


@lru_cache(maxsize=1)
def _unsupported_sampling_params() -> frozenset[str]:
    """Sampling parameters the installed SDK's message methods no longer accept.

    Derived from the signature of `Messages.create`, which `beta.messages` and
    the async resources track. No client is instantiated, so this is safe to
    call during payload construction without resolved credentials.
    """
    try:
        from anthropic.resources.messages import Messages

        accepted = inspect.signature(Messages.create).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        # The SDK moved the resource or the signature is unreadable. Guess from
        # the major version rather than failing: a wrong guess costs a TypeError
        # on the first request, an exception here breaks every import.
        major = _sdk_major_version()
        logger.warning(
            "Could not introspect the anthropic SDK's `Messages.create` "
            "signature; falling back to its major version (%r) to decide "
            "whether `temperature`/`top_p`/`top_k` are still accepted.",
            major,
        )
        if major is not None and major >= 1:
            return frozenset(_SAMPLING_PARAMS)
        return frozenset()

    return frozenset(p for p in _SAMPLING_PARAMS if p not in accepted)


def _route_unsupported_sampling_params(payload: dict[str, Any]) -> dict[str, Any]:
    """Move sampling params the installed SDK rejects into `extra_body`.

    A no-op on `anthropic<1`, where the message methods still accept them as
    named arguments. Values the caller already set in `extra_body` win, since
    that is the SDK's documented escape hatch and a deliberate choice.

    Args:
        payload: Request payload, modified in place.

    Returns:
        The same payload.
    """
    unsupported = _unsupported_sampling_params()
    if not unsupported:
        return payload

    extra_body = payload.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, Mapping):
        # A value the SDK would reject anyway. Leave the payload untouched
        # rather than relocating params into something that cannot hold them.
        return payload

    relocated = {k: payload.pop(k) for k in unsupported if k in payload}
    if not relocated:
        return payload

    payload["extra_body"] = {**relocated, **(extra_body or {})}
    return payload


async def _aparse(raw_response: Any) -> Any:
    """Parse a raw SDK response from the async client.

    `anthropic>=1` returns `AsyncAPIResponse`, whose `parse()` is a coroutine;
    `anthropic<1` returned `LegacyAPIResponse`, whose `parse()` was
    synchronous.
    """
    parsed = raw_response.parse()
    if inspect.isawaitable(parsed):
        return await parsed
    return parsed
