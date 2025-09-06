"""Test text splitting functionality using NLTK and Spacy based sentence splitters."""

import re

import nltk
import pytest
from langchain_core.documents import Document

from langchain_text_splitters.nltk import NLTKTextSplitter
from langchain_text_splitters.spacy import SpacyTextSplitter


def setup_module() -> None:
    nltk.download("punkt_tab")


@pytest.fixture
def spacy() -> None:
    spacy = pytest.importorskip("spacy")

    # Check if en_core_web_sm model is available
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip(
            "en_core_web_sm model not installed. Install with: "
            "uv add --group test_integration "
            "https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        )


def test_nltk_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Got a larger chunk overlap (4) than chunk size (2), should be smaller."
        ),
    ):
        NLTKTextSplitter(chunk_size=2, chunk_overlap=4)


@pytest.mark.usefixtures("spacy")
def test_spacy_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Got a larger chunk overlap (4) than chunk size (2), should be smaller."
        ),
    ):
        SpacyTextSplitter(chunk_size=2, chunk_overlap=4)


def test_nltk_text_splitter() -> None:
    """Test splitting by sentence using NLTK."""
    text = "This is sentence one. And this is sentence two."
    separator = "|||"
    splitter = NLTKTextSplitter(separator=separator)
    output = splitter.split_text(text)
    expected_output = [f"This is sentence one.{separator}And this is sentence two."]
    assert output == expected_output


@pytest.mark.usefixtures("spacy")
@pytest.mark.parametrize("pipeline", ["sentencizer", "en_core_web_sm"])
def test_spacy_text_splitter(pipeline: str) -> None:
    """Test splitting by sentence using Spacy."""
    text = "This is sentence one. And this is sentence two."
    separator = "|||"
    splitter = SpacyTextSplitter(separator=separator, pipeline=pipeline)
    output = splitter.split_text(text)
    expected_output = [f"This is sentence one.{separator}And this is sentence two."]
    assert output == expected_output


@pytest.mark.usefixtures("spacy")
@pytest.mark.parametrize("pipeline", ["sentencizer", "en_core_web_sm"])
def test_spacy_text_splitter_strip_whitespace(pipeline: str) -> None:
    """Test splitting by sentence using Spacy."""
    text = "This is sentence one. And this is sentence two."
    separator = "|||"
    splitter = SpacyTextSplitter(
        separator=separator, pipeline=pipeline, strip_whitespace=False
    )
    output = splitter.split_text(text)
    expected_output = [f"This is sentence one. {separator}And this is sentence two."]
    assert output == expected_output


def test_nltk_text_splitter_args() -> None:
    """Test invalid arguments for NLTKTextSplitter."""
    with pytest.raises(
        ValueError, match="When use_span_tokenize is True, separator should be ''"
    ):
        NLTKTextSplitter(
            chunk_size=80,
            chunk_overlap=0,
            separator="\n\n",
            use_span_tokenize=True,
        )


def test_nltk_text_splitter_with_add_start_index() -> None:
    splitter = NLTKTextSplitter(
        chunk_size=80,
        chunk_overlap=0,
        separator="",
        use_span_tokenize=True,
        add_start_index=True,
    )
    txt = (
        "Innovation drives our success.        "
        "Collaboration fosters creative solutions. "
        "Efficiency enhances data management."
    )
    docs = [Document(txt)]
    chunks = splitter.split_documents(docs)
    assert len(chunks) == 2
    for chunk in chunks:
        s_i = chunk.metadata["start_index"]
        assert chunk.page_content == txt[s_i : s_i + len(chunk.page_content)]
