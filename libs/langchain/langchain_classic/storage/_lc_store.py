"""Create a key-value store for any langchain serializable object."""

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore, ByteStore

from langchain_classic.storage.encoder_backed import EncoderBackedStore


def _dump_as_bytes(obj: Serializable) -> bytes:
    """Return a bytes representation of a `Document`."""
    return dumps(obj).encode("utf-8")


def _dump_document_as_bytes(obj: Any) -> bytes:
    """Return a bytes representation of a `Document`."""
    if not isinstance(obj, Document):
        msg = "Expected a Document instance"
        raise TypeError(msg)
    return dumps(obj).encode("utf-8")


def _load_document_from_bytes(serialized: bytes) -> Document:
    """Return a document from a bytes representation."""
    obj = loads(serialized.decode("utf-8"))
    if not isinstance(obj, Document):
        msg = f"Expected a Document instance. Got {type(obj)}"
        raise TypeError(msg)
    return obj


def _load_from_bytes(serialized: bytes) -> Serializable:
    """Return a document from a bytes representation."""
    return loads(serialized.decode("utf-8"))


def _identity(x: str) -> str:
    """Return the same object."""
    return x


# PUBLIC API


def create_lc_store(
    store: ByteStore,
    *,
    key_encoder: Callable[[str], str] | None = None,
) -> BaseStore[str, Serializable]:
    """Create a store for LangChain serializable objects from a bytes store.

    Args:
        store: A bytes store to use as the underlying store.
        key_encoder: A function to encode keys; if `None` uses identity function.

    Returns:
        A key-value store for `Document` objects.
    """
    return EncoderBackedStore(
        store,
        key_encoder or _identity,
        _dump_as_bytes,
        _load_from_bytes,
    )


def create_kv_docstore(
    store: ByteStore,
    *,
    key_encoder: Callable[[str], str] | None = None,
) -> BaseStore[str, Document]:
    """Create a store for langchain `Document` objects from a bytes store.

    This store does run time type checking to ensure that the values are
    `Document` objects.

    Args:
        store: A bytes store to use as the underlying store.
        key_encoder: A function to encode keys; if `None`, uses identity function.

    Returns:
        A key-value store for `Document` objects.
    """
    return EncoderBackedStore(
        store,
        key_encoder or _identity,
        _dump_document_as_bytes,
        _load_document_from_bytes,
    )
