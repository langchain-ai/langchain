"""Create a key-value store for any langchain serializable object."""
from typing import Callable

from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.load.serializable import Serializable
from langchain.schema import BaseStore
from langchain.storage import EncoderBackedStore


def _dump_as_bytes(obj: Serializable) -> bytes:
    """Return a bytes representation of a document."""
    return dumps(obj).encode("utf-8")


def _load_from_bytes(serialized: bytes) -> Serializable:
    """Return a document from a bytes representation."""
    return loads(serialized.decode("utf-8"))


def _identity(x: str) -> str:
    """Return the same object."""
    return x


# PUBLIC API


def create_lc_store(
    store: BaseStore[str, bytes],
    *,
    key_encoder: Callable[[str], str] = _identity,
) -> BaseStore[str, Serializable]:
    """Create a docstore from a bytes base store and encoders/decoders.

    See the storage module for more information on available bytes stores.

    Args:
        store: A bytes store to use as the underlying store.
        key_encoder: A function to encode keys; defaults to the identity function.

    Returns:
        A key-value store for documents.
    """
    return EncoderBackedStore(
        store,
        key_encoder,
        _dump_as_bytes,
        _load_from_bytes,
    )
