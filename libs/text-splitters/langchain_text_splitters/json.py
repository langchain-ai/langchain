"""JSON text splitter."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from collections.abc import Callable


class RecursiveJsonSplitter:
    """Splits JSON data into smaller, structured chunks while preserving hierarchy.

    This class provides methods to split JSON data into smaller dictionaries or
    JSON-formatted strings based on configurable maximum and minimum chunk sizes.
    It supports nested JSON structures, optionally converts lists into dictionaries
    for better chunking, and allows the creation of document objects for further use.
    """

    max_chunk_size: int = 2000
    """The maximum size for each chunk."""

    min_chunk_size: int = 1800
    """The minimum size for each chunk, derived from `max_chunk_size` if not
    explicitly provided.
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int | None = None,
        *,
        length_function: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        """Initialize the chunk size configuration for text processing.

        This constructor sets up the maximum and minimum chunk sizes, ensuring that
        the `min_chunk_size` defaults to a value slightly smaller than the
        `max_chunk_size` if not explicitly provided.

        Args:
            max_chunk_size: The maximum size for a chunk.
            min_chunk_size: The minimum size for a chunk.

                If `None`, defaults to the maximum chunk size minus 200, with a lower
                bound of 50.
            length_function: Function that measures the size of a chunk.

                If `None`, chunks are measured by the length of their serialized JSON
                representation. Pass a custom callable to measure chunks in different
                units, such as tokens.
        """
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else max(max_chunk_size - 200, 50)
        )
        self._length_function: Callable[[dict[str, Any]], int] = (
            length_function if length_function is not None else self._json_size
        )

    @staticmethod
    def _json_size(data: dict[str, Any]) -> int:
        """Calculate the size of the serialized JSON object."""
        return len(json.dumps(data))

    @staticmethod
    def _set_nested_dict(
        d: dict[str, Any],
        path: list[str],
        value: Any,  # noqa: ANN401
    ) -> None:
        """Set a value in a nested dictionary based on the given path."""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _list_to_dict_preprocessing(
        self,
        data: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
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

    def _try_set_nested_dict(
        self,
        chunk: dict[str, Any],
        path: list[str],
        value: Any,  # noqa: ANN401
    ) -> bool:
        """Add a value to a chunk, keeping the chunk within `max_chunk_size`.

        The candidate chunk is measured *after* the value has been inserted rather
        than by adding the size of the value to the size of the chunk. Sizes are not
        necessarily additive: serialized JSON shares the punctuation and the path
        prefixes of the keys it already holds, and tokenizers may merge or split
        tokens at the join. Measuring the merged chunk keeps the size limit accurate
        for any `length_function`.

        Args:
            chunk: The chunk to add the value to. Mutated only when the value fits.
            path: The path of keys at which to place the value.
            value: The value to place at `path`.

        Returns:
            `True` if the value was added, `False` if it would overflow the chunk,
            in which case `chunk` is left unchanged.
        """
        node = chunk
        # The shallowest container created by this insertion, if any. Discarding it
        # is enough to undo the whole insertion.
        created: tuple[dict[str, Any], str] | None = None
        for key in path[:-1]:
            if key not in node:
                if created is None:
                    created = (node, key)
                node[key] = {}
            node = node[key]
        node[path[-1]] = value

        if self._length_function(chunk) <= self.max_chunk_size:
            return True

        if created is not None:
            parent, key = created
            del parent[key]
        else:
            del node[path[-1]]
        return False

    def _json_split(
        self,
        data: Any,  # noqa: ANN401
        current_path: list[str] | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Split json into maximum size dictionaries while preserving structure."""
        current_path = current_path or []
        chunks = chunks if chunks is not None else [{}]
        if isinstance(data, dict) and data:
            for key, value in data.items():
                new_path = [*current_path, key]
                if self._try_set_nested_dict(chunks[-1], new_path, value):
                    continue

                if self._length_function(chunks[-1]) >= self.min_chunk_size:
                    # Chunk is big enough, start a new chunk
                    chunks.append({})

                # Iterate
                self._json_split(value, new_path, chunks)
        # Handle leaf values and empty dicts
        elif current_path and not self._try_set_nested_dict(
            chunks[-1], current_path, data
        ):
            # The value cannot be split any further. Give it a fresh chunk so it only
            # overflows on its own, rather than pushing a partially filled chunk over
            # the limit.
            if chunks[-1]:
                chunks.append({})
            self._set_nested_dict(chunks[-1], current_path, data)
        return chunks

    def split_json(
        self,
        json_data: dict[str, Any],
        convert_lists: bool = False,  # noqa: FBT001,FBT002
    ) -> list[dict[str, Any]]:
        """Splits JSON into a list of JSON chunks.

        Args:
            json_data: The JSON data to be split.
            convert_lists: Whether to convert lists in the JSON to dictionaries
                before splitting.

        Returns:
            A list of JSON chunks.

        Raises:
            TypeError: If `json_data` is not a dict and cannot be converted to
                one. `None` returns an empty list rather than raising. A
                top-level list is only accepted when `convert_lists` is `True`.
        """
        is_list_input = isinstance(json_data, list)

        if convert_lists:
            json_data = self._list_to_dict_preprocessing(json_data)

        if json_data is not None and not isinstance(json_data, dict):
            msg = f"json_data must be a dict, got {type(json_data).__name__}."
            if is_list_input and not convert_lists:
                msg += " Top-level lists can be split by passing convert_lists=True."
            raise TypeError(msg)

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
        """Splits JSON into a list of JSON formatted strings.

        Args:
            json_data: The JSON data to be split.
            convert_lists: Whether to convert lists in the JSON to dictionaries
                before splitting.
            ensure_ascii: Whether to ensure ASCII encoding in the JSON strings.

        Returns:
            A list of JSON formatted strings.
        """
        chunks = self.split_json(json_data=json_data, convert_lists=convert_lists)

        # Convert to string
        return [json.dumps(chunk, ensure_ascii=ensure_ascii) for chunk in chunks]

    def create_documents(
        self,
        texts: list[dict[str, Any]],
        convert_lists: bool = False,  # noqa: FBT001,FBT002
        ensure_ascii: bool = True,  # noqa: FBT001,FBT002
        metadatas: list[dict[Any, Any]] | None = None,
    ) -> list[Document]:
        """Create a list of `Document` objects from a list of json objects (`dict`).

        Args:
            texts: A list of JSON data to be split and converted into documents.
            convert_lists: Whether to convert lists to dictionaries before splitting.
            ensure_ascii: Whether to ensure ASCII encoding in the JSON strings.
            metadatas: Optional list of metadata to associate with each document.

        Returns:
            A list of `Document` objects.
        """
        metadatas_ = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(
                json_data=text, convert_lists=convert_lists, ensure_ascii=ensure_ascii
            ):
                metadata = copy.deepcopy(metadatas_[i])
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents
