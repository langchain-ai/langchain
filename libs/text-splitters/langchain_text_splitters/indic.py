"""Indic text splitter."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class IndicTextSplitter(RecursiveCharacterTextSplitter):
    """Splitting text for Indic languages (Hindi, Marathi, Nepali, Sanskrit, Bengali, etc.).

    These languages often use the Poorna Viram (।) as a full stop/sentence terminator.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Indic text splitter."""
        separators = kwargs.pop("separators", None)
        if separators is None:
            separators = [
                "\n\n",
                "\n",
                "।",  # Poorna Viram (Devanagari/Bengali full stop)
                "॥",  # Deergha Viram (Double full stop)
                ".",
                " ",
                "",
            ]
        super().__init__(separators=separators, **kwargs)
