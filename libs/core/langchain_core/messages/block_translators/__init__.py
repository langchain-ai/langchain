"""Derivations of standard content blocks from provider content."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.messages import content as types

# Provider to translator mapping
PROVIDER_TRANSLATORS: dict[str, dict[str, Callable[..., list[types.ContentBlock]]]] = {}


def register_translator(
    provider: str,
    translate_content: Callable[[AIMessage], list[types.ContentBlock]],
    translate_content_chunk: Callable[[AIMessageChunk], list[types.ContentBlock]],
) -> None:
    """Register content translators for a provider.

    Args:
        provider: The model provider name (e.g. ``'openai'``, ``'anthropic'``).
        translate_content: Function to translate ``AIMessage`` content.
        translate_content_chunk: Function to translate ``AIMessageChunk`` content.
    """
    PROVIDER_TRANSLATORS[provider] = {
        "translate_content": translate_content,
        "translate_content_chunk": translate_content_chunk,
    }


def get_translator(
    provider: str,
) -> dict[str, Callable[..., list[types.ContentBlock]]] | None:
    """Get the translator functions for a provider.

    Args:
        provider: The model provider name.

    Returns:
        Dictionary with ``'translate_content'`` and ``'translate_content_chunk'``
        functions, or None if no translator is registered for the provider.
    """
    return PROVIDER_TRANSLATORS.get(provider)


def _auto_register_translators() -> None:
    """Automatically register all available block translators."""
    import contextlib
    import importlib
    import pkgutil
    from pathlib import Path

    package_path = Path(__file__).parent

    # Discover all sub-modules
    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name

        # Skip the __init__ module and any private modules
        if module_name.startswith("_"):
            continue

        if module_info.ispkg:
            # For subpackages, discover their submodules
            subpackage_path = package_path / module_name
            for submodule_info in pkgutil.iter_modules([str(subpackage_path)]):
                submodule_name = submodule_info.name
                if not submodule_name.startswith("_"):
                    with contextlib.suppress(ImportError, AttributeError):
                        importlib.import_module(
                            f".{module_name}.{submodule_name}", package=__name__
                        )
        else:
            # Import top-level translator modules
            with contextlib.suppress(ImportError, AttributeError):
                importlib.import_module(f".{module_name}", package=__name__)


_auto_register_translators()
