"""Chonkie text splitter."""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from langchain_text_splitters.base import TextSplitter

try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from chonkie.chunker.base import BaseChunker  # type: ignore[import]
    from chonkie.pipeline import (  # type: ignore[import]
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



class ChonkieTextSplitter(TextSplitter):
    """Splitting text using Chonkie package."""

    valid_chunker_aliases: list[str] = CHUNKERS

    def __init__(
        self,
        chunker: str | BaseChunker = "recursive",
        **kwargs: Any,
    ) -> None:
        """Initialize the Chonkie text splitter.

        Args:
            chunker: The chunker to use for splitting text.

        Raises:
            ImportError: If Chonkie is not installed.
        """
        super().__init__(**kwargs)
        if not _HAS_CHONKIE:
            msg = """
                Chonkie is not installed, please install it with
                `pip install chonkie`
                """
            raise ImportError(msg)
        if isinstance(chunker, str):
            # flexible approach to pull chunker classes based on their alias
            ChunkingClass = ComponentRegistry.get_chunker(chunker).component_class  # noqa: N806
            self.chunker = ChunkingClass(**kwargs)
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
