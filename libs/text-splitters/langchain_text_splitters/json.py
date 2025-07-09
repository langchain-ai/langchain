from __future__ import annotations

import copy
import json
from typing import Any, Optional

from langchain_core.documents import Document


class RecursiveJsonSplitter:
    """Splits JSON data into smaller, structured chunks while preserving hierarchy.

    This class provides methods to split JSON data into smaller dictionaries or
    JSON-formatted strings based on configurable maximum and minimum chunk sizes.
    It supports nested JSON structures, optionally converts lists into dictionaries
    for better chunking, and allows the creation of document objects for further use.

    Attributes:
        max_chunk_size (int): The maximum size for each chunk. Defaults to 2000.
        min_chunk_size (int): The minimum size for each chunk, derived from
            `max_chunk_size` if not explicitly provided.
    """

    def __init__(
        self, max_chunk_size: int = 2000, min_chunk_size: Optional[int] = None
    ):
        """Initialize the chunk size configuration for text processing.

        This constructor sets up the maximum and minimum chunk sizes, ensuring that
        the `min_chunk_size` defaults to a value slightly smaller than the
        `max_chunk_size` if not explicitly provided.

        Args:
            max_chunk_size (int): The maximum size for a chunk. Defaults to 2000.
            min_chunk_size (Optional[int]): The minimum size for a chunk. If None,
                defaults to the maximum chunk size minus 200, with a lower bound of 50.

        Attributes:
            max_chunk_size (int): The configured maximum size for each chunk.
            min_chunk_size (int): The configured minimum size for each chunk, derived
                from `max_chunk_size` if not explicitly provided.
        """
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else max(max_chunk_size - 200, 50)
        )

    @staticmethod
    def _json_size(data: dict[str, Any]) -> int:
        """Calculate the size of the serialized JSON object."""
        return len(json.dumps(data))

    @staticmethod
    def _set_nested_dict(d: dict[str, Any], path: list[str], value: Any) -> None:
        """Set a value in a nested dictionary based on the given path."""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _list_to_dict_preprocessing(self, data: Any) -> Any:
        if isinstance(data, dict):
            # Process each key-value pair in the dictionary
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        if isinstance(data, list):
            # Convert the list to a dictionary with index-based keys
            return {
                str(i): self._list_to_dict_preprocessing(item)
                for i, item in enumerate(data)
            }
        # Base case: the item is neither a dict nor a list, so return it unchanged
        return data

    def _json_split(
        self,
        data: dict[str, Any],
        current_path: Optional[list[str]] = None,
        chunks: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Split json into maximum size dictionaries while preserving structure."""
        current_path = current_path or []
        chunks = chunks if chunks is not None else [{}]
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = [*current_path, key]
                chunk_size = self._json_size(chunks[-1])
                size = self._json_size({key: value})
                remaining = self.max_chunk_size - chunk_size

                if size < remaining:
                    # Add item to current chunk
                    self._set_nested_dict(chunks[-1], new_path, value)
                else:
                    if chunk_size >= self.min_chunk_size:
                        # Chunk is big enough, start a new chunk
                        chunks.append({})

                    # Iterate
                    self._json_split(value, new_path, chunks)
        else:
            # handle single item
            self._set_nested_dict(chunks[-1], current_path, data)
        return chunks

    def split_json(
        self,
        json_data: dict[str, Any],
        convert_lists: bool = False,  # noqa: FBT001,FBT002
    ) -> list[dict[str, Any]]:
        """Splits JSON into a list of JSON chunks."""
        if convert_lists:
            chunks = self._json_split(self._list_to_dict_preprocessing(json_data))
        else:
            chunks = self._json_split(json_data)

        # Remove the last chunk if it's empty
        if not chunks[-1]:
            chunks.pop()
        return chunks

    def split_text(
        self,
        json_data: dict[str, Any],
        convert_lists: bool = False,  # noqa: FBT001,FBT002
        ensure_ascii: bool = True,  # noqa: FBT001,FBT002
    ) -> list[str]:
        """Splits JSON into a list of JSON formatted strings."""
        chunks = self.split_json(json_data=json_data, convert_lists=convert_lists)

        # Convert to string
        return [json.dumps(chunk, ensure_ascii=ensure_ascii) for chunk in chunks]

    def create_documents(
        self,
        texts: list[dict[str, Any]],
        convert_lists: bool = False,  # noqa: FBT001,FBT002
        ensure_ascii: bool = True,  # noqa: FBT001,FBT002
        metadatas: Optional[list[dict[Any, Any]]] = None,
    ) -> list[Document]:
        """Create documents from a list of json objects (Dict)."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(
                json_data=text, convert_lists=convert_lists, ensure_ascii=ensure_ascii
            ):
                metadata = copy.deepcopy(_metadatas[i])
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents
