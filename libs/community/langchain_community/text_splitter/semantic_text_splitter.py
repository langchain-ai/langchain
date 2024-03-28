from __future__ import annotations

from typing import Any, List

from langchain_text_splitters import TextSplitter


class SemanticTiktokenTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(
        self,
        *,
        max_characters: int = 200,
        model_name: str = "gpt-3.5-turbo",
        trim_chunks: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self.max_characters = max_characters
        self.trim_chunks = trim_chunks
        self.model_name = model_name

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        try:
            from semantic_text_splitter import TiktokenTextSplitter
        except ImportError as e:
            raise ImportError(
                "Unable to import from semantic-text-splitter."
                "Please install semantic-text-splitter with "
                "`pip install -U semantic-text-splitter`."
            ) from e
        # Optionally can also have the splitter not trim whitespace for you
        splitter = TiktokenTextSplitter(self.model_name, trim_chunks=self.trim_chunks)
        return splitter.chunks(text, self.max_characters)


class SemanticCharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(
        self, *, max_characters: int = 200, trim_chunks: bool = False, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self.max_characters = max_characters
        self.trim_chunks = trim_chunks

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        try:
            from semantic_text_splitter import CharacterTextSplitter
        except ImportError as e:
            raise ImportError(
                "Unable to import from semantic-text-splitter."
                "Please install semantic-text-splitter with "
                "`pip install -U semantic-text-splitter`."
            ) from e
        splitter = CharacterTextSplitter(trim_chunks=self.trim_chunks)
        return splitter.chunks(text, self.max_characters)
