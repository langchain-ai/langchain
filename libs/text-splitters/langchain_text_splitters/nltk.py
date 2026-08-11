"""NLTK text splitter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from langchain_text_splitters.base import TextSplitter

if TYPE_CHECKING:
    from collections.abc import Callable


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
        """Initialize the NLTK splitter.

        Args:
            separator: The separator to use when combining splits.
            language: The language to use.
            use_span_tokenize: Whether to use `span_tokenize` instead of
                `sent_tokenize`.

        Raises:
            ImportError: If NLTK is not installed.
            ValueError: If `use_span_tokenize` is `True` and separator is not `''`.
        """
        super().__init__(**kwargs)
        self._separator = separator
        if use_span_tokenize and self._separator:
            msg = "When use_span_tokenize is True, separator should be ''"
            raise ValueError(msg)
        try:
            import nltk  # noqa: PLC0415,F401
        except ImportError as err:
            msg = "NLTK is not installed, please install it with `pip install nltk`."
            raise ImportError(msg) from err
        if use_span_tokenize:
            self._tokenizer = self._span_tokenizer(language)
        else:
            self._tokenizer = self._sent_tokenizer(language)

    @staticmethod
    def _sent_tokenizer(language: str) -> Callable[[str], list[str]]:
        import nltk  # noqa: PLC0415

        return lambda text: nltk.tokenize.sent_tokenize(text, language)

    @staticmethod
    def _span_tokenizer(language: str) -> Callable[[str], list[str]]:
        import nltk  # noqa: PLC0415

        tokenizer = nltk.tokenize._get_punkt_tokenizer(language)  # noqa: SLF001

        def _tokenize(text: str) -> list[str]:
            spans = list(tokenizer.span_tokenize(text))
            splits = []
            for i, (start, end) in enumerate(spans):
                if i > 0:
                    prev_end = spans[i - 1][1]
                    sentence = text[prev_end:start] + text[start:end]
                else:
                    sentence = text[start:end]
                splits.append(sentence)
            return splits

        return _tokenize

    @override
    def split_text(self, text: str) -> list[str]:
        # First we naively split the large input into a bunch of smaller ones.
        splits = self._tokenizer(text)
        return self._merge_splits(splits, self._separator)
