"""Derivations of standard content blocks from Google (VertexAI) content."""

from langchain_core.messages.block_translators.google_genai import (
    translate_content,
    translate_content_chunk,
)


def _register_google_vertexai_translator() -> None:
    """Register the Google (VertexAI) translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("google_vertexai", translate_content, translate_content_chunk)


_register_google_vertexai_translator()
