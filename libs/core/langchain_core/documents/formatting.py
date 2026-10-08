"""Document formatting utilities for RAG and LLM prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document


def format_context(
    docs: Sequence[Document],
    *,
    separator: str = "\n\n---\n\n",
) -> str:
    r"""Format a sequence of documents into a single context string for LLM prompts.

    Handles mixed document formats gracefully:
    - Safely extracts 'file_name' or 'source' without KeyError.
    - Handles page numbers: converts 0-indexed integer pages to 1-indexed.
    - Preserves custom page labels (e.g. roman numerals like 'iv').
    - Omits page numbers for unpaged files (CSV, TXT, JSON, Markdown).
    - Divides chunks with a clear separator (default: '\n\n---\n\n').

    Args:
        docs: A sequence of Document objects.
        separator: Separator string between formatted document chunks.
            Defaults to '\n\n---\n\n'.

    Returns:
        Formatted context string.

    Example:
        ```python
        from langchain_core.documents import Document, format_context

        docs = [
            Document(
                page_content="Quarterly revenue grew by 20%.",
                metadata={"source": "report.pdf", "page": 0},
            ),
            Document(
                page_content="id,status\n1,ok",
                metadata={"source": "data.csv"},
            ),
        ]
        context = format_context(docs)
        ```
    """
    formatted_chunks: list[str] = []
    for doc in docs:
        file_name = (
            doc.metadata.get("file_name") or doc.metadata.get("source") or "Document"
        )

        p = doc.metadata.get("page_label")
        if p is None:
            p = doc.metadata.get("page")

        page = p + 1 if isinstance(p, int) else p

        if page is not None:
            header = f"[Source: {file_name} | Page: {page}]"
        else:
            header = f"[Source: {file_name}]"
        formatted_chunks.append(f"{header}\n{doc.page_content}")

    return separator.join(formatted_chunks)
