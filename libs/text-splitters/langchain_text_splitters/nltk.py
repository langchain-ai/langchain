from __future__ import annotations

from typing import Any, List

from langchain_text_splitters.base import TextSplitter


class NLTKTextSplitter(TextSplitter):
    """Splitting text using NLTK package."""

    def __init__(
        self,
        separator: str = "\n\n",
        language: str = "english",
        *,
        use_span_tokenize: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the NLTK splitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._language = language
        self._use_span_tokenize = use_span_tokenize
        if self._use_span_tokenize and self._separator != "":
            raise ValueError("When use_span_tokenize is True, separator should be ''")
        try:
            import nltk

            if self._use_span_tokenize:
                self._tokenizer = nltk.tokenize._get_punkt_tokenizer(self._language)
            else:
                self._tokenizer = nltk.tokenize.sent_tokenize
        except ImportError:
            raise ImportError(
                "NLTK is not installed, please install it with `pip install nltk`."
            )

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        if self._use_span_tokenize:
            spans = list(self._tokenizer.span_tokenize(text))
            splits = []
            for i, (start, end) in enumerate(spans):
                if i > 0:
                    prev_end = spans[i - 1][1]
                    sentence = text[prev_end:start] + text[start:end]
                else:
                    sentence = text[start:end]
                splits.append(sentence)
        else:
            splits = self._tokenizer(text, language=self._language)
        return self._merge_splits(splits, self._separator)
