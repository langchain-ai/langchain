"""Module for serialization code.

This code will likely be replaced by Nuno's serialization method.
"""
import json
from json import JSONEncoder, JSONDecodeError
from uuid import UUID

from langchain.schema import Document


class UUIDEncoder(JSONEncoder):
    """Will either be replaced by Nuno's serialization method or something else.

    Potentially there will be no serialization for a document object since
    the document can be broken into 2 pieces:

    * the content -> saved on disk or in database
    * the metadata -> saved in metadata store

    It may not make sense to keep the metadata together with the document
    for the persistence.
    """

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)  # Convert UUID to string
        return super().default(obj)


# PUBLIC API


def serialize_document(document: Document) -> str:
    """Serialize the given document to a string."""
    try:
        # Serialize only the content.
        # Metadata always stored separately.
        return json.dumps(document.page_content)
    except JSONDecodeError:
        raise ValueError(f"Could not serialize document with ID: {document.uid}")


def deserialize_document(serialized_document: str) -> Document:
    """Deserialize the given document from a string."""
    return Document(
        page_content=json.loads(serialized_document),
    )
