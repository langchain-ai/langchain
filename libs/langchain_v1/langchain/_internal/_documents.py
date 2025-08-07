"""Internal document utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def format_document_xml(doc: Document) -> str:
    """Format a document as XML-like structure for LLM consumption.

    Args:
        doc: Document to format

    Returns:
        Document wrapped in XML tags:
            <document>
                <id>...</id>
                <content>...</content>
                <metadata>...</metadata>
            </document>

    Note:
        Does not generate valid XML or escape special characters.
        Intended for semi-structured LLM input only.
    """
    id_str = f"<id>{doc.id}</id>" if doc.id is not None else "<id></id>"
    metadata_str = ""
    if doc.metadata:
        metadata_items = [f"{k}: {v!s}" for k, v in doc.metadata.items()]
        metadata_str = f"<metadata>{', '.join(metadata_items)}</metadata>"
    return (
        f"<document>{id_str}"
        f"<content>{doc.page_content}</content>"
        f"{metadata_str}"
        f"</document>"
    )
