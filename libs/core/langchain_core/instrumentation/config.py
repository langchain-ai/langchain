"""Global instrumentation provider registry.

Provides `set_instrumentation_provider` and `get_instrumentation_provider`
for configuring the active observability backend at application startup.

The provider is stored as a module-level singleton (not a ContextVar) because
it represents a process-wide configuration — similar to how logging handlers
are configured once at startup, not per-request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.instrumentation.noop import NoopProvider

if TYPE_CHECKING:
    from langchain_core.instrumentation.provider import InstrumentationProvider

logger = logging.getLogger(__name__)

_provider: InstrumentationProvider = NoopProvider()


def get_instrumentation_provider() -> InstrumentationProvider:
    """Return the currently active instrumentation provider.

    Returns `NoopProvider` if no provider has been configured.

    Returns:
        The active instrumentation provider.
    """
    return _provider


def set_instrumentation_provider(provider: InstrumentationProvider) -> None:
    """Set the global instrumentation provider.

    Should be called once at application startup, before any LangChain
    operations are invoked.

    Args:
        provider: The provider to use for all subsequent instrumentation.

    Example::

        from langchain_core.instrumentation import set_instrumentation_provider
        from my_company.observability import DatadogProvider

        set_instrumentation_provider(DatadogProvider(service="my-app"))
    """
    global _provider  # noqa: PLW0603
    logger.info("Instrumentation provider set to %s", type(provider).__name__)
    _provider = provider


def reset_instrumentation_provider() -> None:
    """Reset the global provider to `NoopProvider`.

    Primarily useful for testing.
    """
    global _provider  # noqa: PLW0603
    _provider = NoopProvider()


__all__ = [
    "get_instrumentation_provider",
    "reset_instrumentation_provider",
    "set_instrumentation_provider",
]
