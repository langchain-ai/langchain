"""Spacy text splitter."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from langchain_text_splitters.base import TextSplitter

if TYPE_CHECKING:
    # Type ignores needed as long as spacy doesn't support Python 3.14.
    from spacy.language import (  # type: ignore[import-not-found, unused-ignore]
        Language,
    )


class SpacyTextSplitter(TextSplitter):
    """Splitting text using Spacy package.

    Per default, Spacy's `en_core_web_sm` model is used and
    its default max_length is 1000000 (it is the length of maximum character
    this model takes which can be increased for large files). For a faster, but
    potentially less accurate splitting, you can use `pipeline='sentencizer'`.
    """

    def __init__(
        self,
        separator: str = "\n\n",
        pipeline: str = "en_core_web_sm",
        max_length: int = 1_000_000,
        *,
        strip_whitespace: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the spacy text splitter."""
        super().__init__(**kwargs)
        self._tokenizer = _make_spacy_pipeline_for_splitting(
            pipeline, max_length=max_length
        )
        self._separator = separator
        self._strip_whitespace = strip_whitespace

    @override
    def split_text(self, text: str) -> list[str]:
        splits = (
            s.text if self._strip_whitespace else s.text_with_ws
            for s in self._tokenizer(text).sents
        )
        return self._merge_splits(splits, self._separator)


def _make_spacy_pipeline_for_splitting(
    pipeline: str, *, max_length: int = 1_000_000
) -> Language:
    try:
        spacy = cast("Any", import_module("spacy"))
        english_cls = cast("Any", import_module("spacy.lang.en")).English
    except ImportError as err:
        msg = "Spacy is not installed, please install it with `pip install spacy`."
        raise ImportError(msg) from err
    if pipeline == "sentencizer":
        sentencizer: Language = english_cls()
        sentencizer.add_pipe("sentencizer")
    else:
        sentencizer = spacy.load(pipeline, exclude=["ner", "tagger"])
        sentencizer.max_length = max_length
    return sentencizer
