"""Regression tests for regex-separator merge corruption.

See https://github.com/langchain-ai/langchain/issues/39569

When ``is_separator_regex=True`` and ``keep_separator=False``, both
``CharacterTextSplitter`` and ``RecursiveCharacterTextSplitter`` re-join split
pieces using the raw regex pattern string as a *literal* merge separator. For
any consuming pattern (e.g. ``r"\\s+"``) that string is not the text the regex
matched, so the merged chunk is corrupted with literal regex syntax (e.g.
``"AAA\\s+BBB"``) and is no longer a substring of the source document.
"""

from langchain_text_splitters.character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


def test_character_text_splitter_consuming_regex_not_reinserted_on_merge() -> None:
    """Merged chunks must not embed the raw consuming regex pattern."""
    splitter = CharacterTextSplitter(
        separator=r"\s+",
        chunk_size=15,
        chunk_overlap=0,
        keep_separator=False,
        is_separator_regex=True,
    )
    output = splitter.split_text("AAA   BBB\tCCC\n\nDDD    EEE")
    # keep_separator=False drops the separator; the merged chunk must never
    # contain the literal pattern text "\s+".
    assert output == ["AAABBBCCCDDDEEE"]


def test_recursive_character_text_splitter_consuming_regex_not_reinserted() -> None:
    """RecursiveCharacterTextSplitter must not embed the raw regex pattern."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=15,
        chunk_overlap=0,
        separators=[r"\s+"],
        keep_separator=False,
        is_separator_regex=True,
    )
    output = splitter.split_text("one   two\tthree\n\nfour     five")
    assert output == ["onetwothreefour", "five"]


def test_character_text_splitter_raw_string_literal_regex_not_corrupted() -> None:
    """A raw-string regex for a literal (e.g. r"\\n\\n") must not leak.

    ``r"\\n\\n"`` matches two real newlines but its *pattern string* is the
    four characters ``\n\n``. Re-inserting that pattern verbatim would embed
    literal backslashes into the output.
    """
    splitter = CharacterTextSplitter(
        separator=r"\n\n",
        chunk_size=200,
        chunk_overlap=0,
        keep_separator=False,
        is_separator_regex=True,
    )
    assert splitter.split_text("foo\n\nbar") == ["foobar"]


def test_character_text_splitter_literal_regex_separator_still_reinserted() -> None:
    """A truly literal regex separator is still re-inserted on merge."""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=200,
        chunk_overlap=0,
        keep_separator=False,
        is_separator_regex=True,
    )
    assert splitter.split_text("foo\n\nbar") == ["foo\n\nbar"]


def test_character_text_splitter_consuming_regex_keep_separator_true() -> None:
    """keep_separator=True must still preserve the actual matched separator."""
    splitter = CharacterTextSplitter(
        separator=r"\s+",
        chunk_size=200,
        chunk_overlap=0,
        keep_separator=True,
        is_separator_regex=True,
    )
    assert splitter.split_text("AAA   BBB") == ["AAA   BBB"]
