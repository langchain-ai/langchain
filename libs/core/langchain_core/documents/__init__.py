"""Documents module for data retrieval and processing workflows.

This module provides core abstractions for handling data in retrieval-augmented
generation (RAG) pipelines, vector stores, and document processing workflows.

!!! warning "Documents vs. message content"
    This module is distinct from `langchain_core.messages.content`, which provides
    multimodal content blocks for **LLM chat I/O** (text, images, audio, etc. within
    messages).

    **Key distinction:**

    - **Documents** (this module): For **data retrieval and processing workflows**
        - Vector stores, retrievers, RAG pipelines
        - Text chunking, embedding, and semantic search
        - Example: Chunks of a PDF stored in a vector database

    - **Content Blocks** (`messages.content`): For **LLM conversational I/O**
        - Multimodal message content sent to/from models
        - Tool calls, reasoning, citations within chat
        - Example: An image sent to a vision model in a chat message (via
            [`ImageContentBlock`][langchain.messages.ImageContentBlock])

    While both can represent similar data types (text, files), they serve different
    architectural purposes in LangChain applications.
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from .base import Document
    from .compressor import BaseDocumentCompressor
    from .transformers import BaseDocumentTransformer

__all__ = ("BaseDocumentCompressor", "BaseDocumentTransformer", "Document")

_dynamic_imports = {
    "Document": "base",
    "BaseDocumentCompressor": "compressor",
    "BaseDocumentTransformer": "transformers",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
