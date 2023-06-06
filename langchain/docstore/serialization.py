"""Module for serialization code.

This code will likely be replaced by Nuno's serialization method.
"""
import json
from json import JSONEncoder, JSONDecodeError
from uuid import UUID

from langchain.schema import Document


class UUIDEncoder(JSONEncoder):
    """TODO detemine if there's a better solution."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)  # Convert UUID to string
        return super().default(obj)


def serialize_document(document: Document) -> str:
    """Serialize the given document to a string."""
    try:
        return json.dumps(document.dict(), cls=UUIDEncoder)
    except JSONDecodeError:
        raise ValueError(f"Could not serialize document with ID: {document.id}")


def deserialize_document(serialized_document: str) -> Document:
    """Deserialize the given document from a string."""
    return Document.parse_obj(json.loads(serialized_document))
