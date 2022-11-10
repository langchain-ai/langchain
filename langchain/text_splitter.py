"""Functionality for splitting text."""
from abc import abstractmethod
from typing import Iterable, List


class TextSplitter:
    """Interface for splitting text into chunks."""

    def __init__(self, separator: str, chunk_size: int, chunk_overlap: int):
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._separator = separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def _merge_splits(self, splits: Iterable[str]) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            if total > self._chunk_size:
                docs.append(self._separator.join(current_doc))
                while total > self._chunk_overlap:
                    total -= len(current_doc[0])
                    current_doc = current_doc[1:]
            current_doc.append(d)
            total += len(d)
        docs.append(self._separator.join(current_doc))
        return docs


class CharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters."""

    def __init__(
        self, separator: str = "\n\n", chunk_size: int = 4000, chunk_overlap: int = 200
    ):
        """Create a new CharacterTextSplitter."""
        super(CharacterTextSplitter, self).__init__(
            separator, chunk_size, chunk_overlap
        )
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = text.split(self._separator)
        return self._merge_splits(splits)


class NLTKTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at sentences using NLTK."""

    def __init__(
        self, separator: str = "\n\n", chunk_size: int = 4000, chunk_overlap: int = 200
    ):
        """Initialize the NLTK splitter."""
        super(NLTKTextSplitter, self).__init__(separator, chunk_size, chunk_overlap)
        try:
            from nltk.tokenize import sent_tokenize

            self._tokenizer = sent_tokenize
        except ImportError:
            raise ImportError(
                "NLTK is not installed, please install it with `pip install nltk`."
            )

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = self._tokenizer(text)
        return self._merge_splits(splits)


class SpacyTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at sentences using Spacy."""

    def __init__(
        self,
        separator: str = "\n\n",
        pipeline: str = "en_core_web_sm",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ):
        """Initialize the spacy text splitter."""
        super(SpacyTextSplitter, self).__init__(separator, chunk_size, chunk_overlap)
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "Spacy is not installed, please install it with `pip install spacy`."
            )
        self._tokenizer = spacy.load(pipeline)

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits = (str(s) for s in self._tokenizer(text).sents)
        return self._merge_splits(splits)
