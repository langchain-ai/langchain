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


def _register_translators() -> None:
    """Register all translators in langchain-core.

    A unit test ensures all modules in ``block_translators`` are represented here.

    For translators implemented outside langchain-core, they can be registered by
    calling ``register_translator`` from within the integration package.
    """
    from langchain_core.messages.block_translators.anthropic import (  # noqa: PLC0415
        _register_anthropic_translator,
    )
    from langchain_core.messages.block_translators.bedrock import (  # noqa: PLC0415
        _register_bedrock_translator,
    )
    from langchain_core.messages.block_translators.bedrock_converse import (  # noqa: PLC0415
        _register_bedrock_converse_translator,
    )
    from langchain_core.messages.block_translators.google_genai import (  # noqa: PLC0415
        _register_google_genai_translator,
    )
    from langchain_core.messages.block_translators.google_vertexai import (  # noqa: PLC0415
        _register_google_vertexai_translator,
    )
    from langchain_core.messages.block_translators.groq import (  # noqa: PLC0415
        _register_groq_translator,
    )
    from langchain_core.messages.block_translators.ollama import (  # noqa: PLC0415
        _register_ollama_translator,
    )
    from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
        _register_openai_translator,
    )

    _register_bedrock_translator()
    _register_bedrock_converse_translator()
    _register_anthropic_translator()
    _register_google_genai_translator()
    _register_google_vertexai_translator()
    _register_groq_translator()
    _register_ollama_translator()
    _register_openai_translator()


_register_translators()
