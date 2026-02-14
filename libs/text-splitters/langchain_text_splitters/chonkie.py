"""Chonkie text splitter."""

from __future__ import annotations

import copy
from typing import Any

from langchain_core.documents import Document
from typing_extensions import override

from langchain_text_splitters.base import TextSplitter

try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from chonkie.chunker.base import (  # type: ignore[import, unused-ignore]
            BaseChunker,
        )
    from chonkie.pipeline import (  # type: ignore[import, unused-ignore]
        ComponentRegistry,
        ComponentType,
    )

    _HAS_CHONKIE = True
    CHUNKERS = sorted(
        c.alias
        for c in ComponentRegistry.list_components(component_type=ComponentType.CHUNKER)
        if c.alias not in ["table", "slumber"]
    )
except ImportError:
    _HAS_CHONKIE = False
    CHUNKERS = []


class ChonkieTextSplitter(TextSplitter):
    """Splitting text using Chonkie package."""

    valid_chunker_aliases: list[str] = CHUNKERS

    def __init__(
        self,
        chunker: str | BaseChunker = "recursive",
        chunk_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Initialize the Chonkie text splitter.

        Args:
            chunker: The chunker to use for splitting text. Defaults to "recursive".
            chunk_size: The maximum size of chunks to return. Defaults to 1024.
            **kwargs: Additional arguments to pass to the chonkie chunker.

        Raises:
            ImportError: If Chonkie is not installed.
        """
        if "chunk_overlap" not in kwargs:
            kwargs["chunk_overlap"] = 0

        # Separate TextSplitter kwargs from Chonkie kwargs
        ts_args = [
            "chunk_overlap",
            "length_function",
            "keep_separator",
            "add_start_index",
            "strip_whitespace",
        ]
        ts_kwargs = {k: v for k, v in kwargs.items() if k in ts_args}
        super().__init__(chunk_size=chunk_size, **ts_kwargs)

        if not _HAS_CHONKIE:
            msg = """
                Chonkie is not installed, please install it with
                `pip install chonkie`
                """
            raise ImportError(msg)

        if isinstance(chunker, str):
            # flexible approach to pull chunker classes based on their alias
            chunking_class = ComponentRegistry.get_chunker(chunker).component_class

            self.chunker = chunking_class(**kwargs)
        else:
            self.chunker = chunker

    @override
    def split_text(self, text: str) -> list[str]:
        chunks = self.chunker(text)
        if isinstance(chunks, list):
            return [
                chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in chunks
            ]
        return [chunks.text if hasattr(chunks, "text") else str(chunks)]

    @override
    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            chunks = self.chunker(text)
            if not isinstance(chunks, list):
                chunks = [chunks]

            for chunk in chunks:
                metadata = copy.deepcopy(_metadatas[i])
                if hasattr(chunk, "start_index"):
                    metadata["start_index"] = chunk.start_index
                if hasattr(chunk, "end_index"):
                    metadata["end_index"] = chunk.end_index
                if hasattr(chunk, "token_count"):
                    metadata["token_count"] = chunk.token_count

                content = chunk.text if hasattr(chunk, "text") else str(chunk)
                new_doc = Document(page_content=content, metadata=metadata)
                documents.append(new_doc)
        return documents
