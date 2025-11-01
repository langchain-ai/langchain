"""Special handling for LangChain-specific types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .constants import JsonValue

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.load.serializable import Serializable
    from langchain_core.messages.base import BaseMessage


def normalize_langchain_value(value: Any) -> JsonValue | None:
    """Normalize LangChain-specific types to JSON-compatible structures.

    This function handles special serialization for core LangChain types:
    - BaseMessage: Extracts type, content, and relevant metadata
    - Document: Extracts page_content, metadata, and optional id
    - Serializable: Uses .to_json() if available

    Args:
        value: Potentially a LangChain-specific object.

    Returns:
        Normalized JSON-compatible value, or `None` if not a recognized type.
    """
    # Check for BaseMessage
    if _is_base_message(value):
        return _normalize_message(value)

    # Check for Document
    if _is_document(value):
        return _normalize_document(value)

    # Check for Serializable
    if _is_serializable(value):
        return _normalize_serializable(value)

    # Check for Pydantic models
    if _is_pydantic_model(value):
        return _normalize_pydantic(value)

    return None


def _is_base_message(value: Any) -> bool:
    """Check if value is a BaseMessage instance.

    Args:
        value: Value to check.

    Returns:
        `True` if value is a BaseMessage.
    """
    try:
        from langchain_core.messages.base import BaseMessage

        return isinstance(value, BaseMessage)
    except ImportError:
        return False


def _normalize_message(message: BaseMessage) -> dict[str, Any]:
    """Normalize a BaseMessage to a dict.

    Args:
        message: BaseMessage instance to normalize.

    Returns:
        Dict with type, content, and relevant metadata.
    """
    result: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }

    # Include id if present
    if hasattr(message, "id") and message.id:
        result["id"] = message.id

    # Include name if present (for named messages)
    if hasattr(message, "name") and message.name:
        result["name"] = message.name

    # Include additional_kwargs if non-empty
    if hasattr(message, "additional_kwargs") and message.additional_kwargs:
        result["additional_kwargs"] = message.additional_kwargs

    # Include response_metadata if non-empty
    if hasattr(message, "response_metadata") and message.response_metadata:
        result["response_metadata"] = message.response_metadata

    return result


def _is_document(value: Any) -> bool:
    """Check if value is a Document instance.

    Args:
        value: Value to check.

    Returns:
        `True` if value is a Document.
    """
    try:
        from langchain_core.documents import Document

        return isinstance(value, Document)
    except ImportError:
        return False


def _normalize_document(document: Document) -> dict[str, Any]:
    """Normalize a Document to a dict.

    Args:
        document: Document instance to normalize.

    Returns:
        Dict with page_content, metadata, and optional id.
    """
    result: dict[str, Any] = {
        "page_content": document.page_content,
    }

    if document.metadata:
        result["metadata"] = document.metadata

    if hasattr(document, "id") and document.id:
        result["id"] = document.id

    return result


def _is_serializable(value: Any) -> bool:
    """Check if value is a Serializable instance.

    Args:
        value: Value to check.

    Returns:
        `True` if value is Serializable and is_lc_serializable returns `True`.
    """
    try:
        from langchain_core.load.serializable import Serializable

        return isinstance(value, Serializable) and value.is_lc_serializable()
    except ImportError:
        return False


def _normalize_serializable(serializable: Serializable) -> Any:
    """Normalize a Serializable object using its to_json method.

    Args:
        serializable: Serializable instance to normalize.

    Returns:
        Result of calling to_json() on the object.
    """
    return serializable.to_json()


def _is_pydantic_model(value: Any) -> bool:
    """Check if value is a Pydantic model.

    Args:
        value: Value to check.

    Returns:
        `True` if value is a Pydantic BaseModel instance.
    """
    try:
        from pydantic import BaseModel

        return isinstance(value, BaseModel)
    except ImportError:
        return False


def _normalize_pydantic(model: Any) -> dict[str, Any]:
    """Normalize a Pydantic model to a dict.

    Args:
        model: Pydantic BaseModel instance to normalize.

    Returns:
        Dict representation using model_dump.
    """
    return model.model_dump(mode="python", exclude_none=False)
