"""Create a key-value store for any langchain serializable object."""

from typing import Any, Callable, Optional

from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore, ByteStore

from langchain.storage.encoder_backed import EncoderBackedStore


def _dump_as_bytes(obj: Serializable) -> bytes:
    """Return a bytes representation of a document."""
    return dumps(obj).encode("utf-8")


def _dump_document_as_bytes(obj: Any) -> bytes:
    """Return a bytes representation of a document."""
    if not isinstance(obj, Document):
        msg = "Expected a Document instance"
        raise TypeError(msg)
    return dumps(obj).encode("utf-8")


def _load_document_from_bytes(serialized: bytes) -> Document:
    """Return a document from a bytes representation."""
    obj = loads(serialized.decode("utf-8"), allowed_objects=[Document])
    if not isinstance(obj, Document):
        msg = f"Expected a Document instance. Got {type(obj)}"
        raise TypeError(msg)
    return obj


def _load_from_bytes(serialized: bytes) -> Serializable:
    """Return a ``Serializable`` from a bytes representation."""
    # The default allowlist (``'core'``) is unsafe with untrusted input - a
    # tampered byte payload can reconstruct any core class with
    # attacker-controlled kwargs (custom ``base_url``, headers, model name,
    # etc.). The byte store backing this loader must be treated as a trust
    # boundary - see the danger note on ``create_lc_store``. If the store
    # can be written to by anyone you do not already trust, use
    # ``create_kv_docstore`` instead.
    return loads(serialized.decode("utf-8"))


def _identity(x: str) -> str:
    """Return the same object."""
    return x


# PUBLIC API


def create_lc_store(
    store: ByteStore,
    *,
    key_encoder: Optional[Callable[[str], str]] = None,
) -> BaseStore[str, Serializable]:
    """Create a store for langchain serializable objects from a bytes store.

    .. danger::

        Treat the underlying byte store as a trust boundary.

        Reads from this store are deserialized with
        ``langchain_core.load.loads``, which instantiates Python objects
        from the stored payload. The same threat model applies: a payload
        can carry constructor kwargs (custom ``base_url``, headers, model
        name, etc.) that get applied during ``__init__``, so the bytes are
        effectively executable configuration rather than plain data.

        **Never back this store with anything an attacker can write to** -
        for example a shared cache that other tenants can populate, an
        S3 bucket without strict write controls, or a Redis instance
        reused across trust boundaries. A single tampered value will
        instantiate attacker-controlled classes the next time the store
        is read.

        If you cannot guarantee the store is write-restricted to your own
        process, use ``create_kv_docstore`` instead - it pins
        ``allowed_objects=[Document]`` so a tampered value can at worst
        produce a ``Document``, never a chat model or LLM with a
        redirected endpoint.

    Args:
        store: A bytes store to use as the underlying store.
        key_encoder: A function to encode keys; if None uses identity function.

    Returns:
        A key-value store for documents.
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
    key_encoder: Optional[Callable[[str], str]] = None,
) -> BaseStore[str, Document]:
    """Create a store for langchain Document objects from a bytes store.

    This store does run time type checking to ensure that the values are
    Document objects.

    Args:
        store: A bytes store to use as the underlying store.
        key_encoder: A function to encode keys; if None uses identity function.

    Returns:
        A key-value store for documents.
    """
    return EncoderBackedStore(
        store,
        key_encoder or _identity,
        _dump_document_as_bytes,
        _load_document_from_bytes,
    )
