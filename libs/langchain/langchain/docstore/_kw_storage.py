from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.schema import BaseStore, Document
from langchain.storage import EncoderBackedStore


def _dump_as_bytes(obj: Document) -> bytes:
    """Return a bytes representation of a document."""
    return dumps(obj).encode("utf-8")


def _load_from_bytes(serialized: bytes) -> Document:
    """Return a document from a bytes representation."""
    return loads(serialized.decode("utf-8"))


# PUBLIC API


def create_kw_docstore(
    store: BaseStore[str, bytes],
) -> BaseStore[str, Document]:
    """Create a docstore from a bytes base store and encoders/decoders.

    See the storage module for more information on available bytes stores.

    Args:
        store: The base store to wrap.

    Returns:
        A key-value store for documents.
    """
    return EncoderBackedStore(
        store=store,
        key_encoder=lambda key: key,
        value_serializer=_dump_as_bytes,
        value_deserializer=_load_from_bytes,
    )
