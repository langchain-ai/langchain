from __future__ import annotations

from typing import Any, List

from langchain_text_splitters.base import TextSplitter
from nltk.tokenize import sent_tokenize

class NLTKTextSplitter(TextSplitter):
    """Splitting text using NLTK package."""

    def __init__(
        self, separator: str = "\n\n", language: str = "english", **kwargs: Any
    ) -> None:
        """Initialize the NLTK splitter."""
        super().__init__(**kwargs)                          
        self._tokenizer = sent_tokenize            
        self._separator = separator
        self._language = language

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = self._tokenizer(text, language=self._language)
        return self._merge_splits(splits, self._separator)

a = NLTKTextSplitter()
print(a.split_text("hello how are you?!"))