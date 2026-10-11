"""Regression tests for TextSplitter start_index metadata.

`create_documents(..., add_start_index=True)` used to recover each chunk with
`str.find` and a guess of `previous_end - chunk_overlap`. That guess moves
backwards when the previous chunk is shorter than `chunk_overlap`, so a later
copy of the same text is reported at the earlier copy's offset.
"""

from __future__ import annotations

import random
import re

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_text_splitters.jsx import JSFrameworkTextSplitter


def _assert_verbatim_and_ordered(text: str, docs: list[Document]) -> None:
    previous = -1
    for doc in docs:
        start = doc.metadata["start_index"]
        assert start >= previous
        assert start >= 0
        assert text[start : start + len(doc.page_content)] == doc.page_content
        previous = start


def test_duplicate_line_shorter_than_overlap_keeps_its_own_start() -> None:
    """The second copy of a short line must not reuse the first copy's index.

    chunk_size=30 and chunk_overlap=20 make the first chunk (20 chars) no
    longer than the overlap, which used to restart the search at offset 0.
    """
    text = "See the table below.\n\nSee the table below.\n\nSummary"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30, chunk_overlap=20, add_start_index=True
    )

    docs = splitter.create_documents([text], metadatas=[{"source": "notes"}])

    assert [doc.page_content for doc in docs] == [
        "See the table below.",
        "See the table below.",
        "Summary",
    ]
    assert [doc.metadata["start_index"] for doc in docs] == [0, 22, 44]
    assert docs[1].metadata == {"source": "notes", "start_index": 22}
    _assert_verbatim_and_ordered(text, docs)


def test_two_chunks_may_still_share_a_start_index() -> None:
    """Overlap can emit two different chunks that both begin at the same index.

    A newline, space, dot, space, dot with chunk_size=4 and chunk_overlap=2
    yields "." and ". .". Forcing the next search to start after the previous
    index would miss the second chunk.
    """
    text = "\n . ."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4, chunk_overlap=2, add_start_index=True
    )

    docs = splitter.create_documents([text])

    assert [doc.page_content for doc in docs] == [".", ". ."]
    assert [doc.metadata["start_index"] for doc in docs] == [2, 2]
    _assert_verbatim_and_ordered(text, docs)


def test_chunk_that_is_a_prefix_of_the_next_chunk_is_not_reused() -> None:
    """A short prefix chunk and the following chunk can start at the same place.

    This is the "chunk chunk" case from the stale report: the second and third
    chunks both start at index 6, and neither is the first "chunk".
    """
    text = "chunk chunk"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5,
        chunk_overlap=5,
        separators=[" ", ""],
        add_start_index=True,
    )

    docs = splitter.create_documents([text])

    assert [doc.page_content for doc in docs] == ["chunk", "chun", "chunk"]
    assert [doc.metadata["start_index"] for doc in docs] == [0, 6, 6]
    _assert_verbatim_and_ordered(text, docs)


def test_character_splitter_overlap_indexes_existing_cases() -> None:
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=7, chunk_overlap=3, add_start_index=True
    )

    docs = splitter.create_documents([text])

    assert docs == [
        Document(page_content="foo bar", metadata={"start_index": 0}),
        Document(page_content="bar baz", metadata={"start_index": 4}),
        Document(page_content="baz 123", metadata={"start_index": 8}),
    ]
    _assert_verbatim_and_ordered(text, docs)


def test_repeated_token_with_zero_overlap_steps_forward() -> None:
    text = "w1 w1 w1 w1 w1 w1 w1 w1 w1"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=6,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    docs = splitter.create_documents([text])

    assert [doc.metadata["start_index"] for doc in docs] == [0, 6, 12, 18, 24]
    _assert_verbatim_and_ordered(text, docs)


def test_strip_whitespace_false_keeps_the_separator_in_the_chunk() -> None:
    text = "See the table below.\n\nSee the table below.\n\nSummary"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=False,
    )

    docs = splitter.create_documents([text])

    assert [(doc.metadata["start_index"], doc.page_content) for doc in docs] == [
        (0, "See the table below."),
        (20, "\n\nSee the table below."),
        (42, "\n\nSummary"),
    ]
    _assert_verbatim_and_ordered(text, docs)


def test_custom_length_function_uses_character_offsets() -> None:
    """`chunk_overlap` is in length-function units, not characters.

    Word-count overlap must not be subtracted from a character cursor.
    """

    def len_in_words(value: str) -> int:
        return len(re.findall(r"\b\w+\b", value))

    text = "test test test test"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2,
        chunk_overlap=1,
        length_function=len_in_words,
        add_start_index=True,
    )

    docs = splitter.create_documents([text])

    assert [doc.page_content for doc in docs] == [
        "test test",
        "test test",
        "test test",
    ]
    assert [doc.metadata["start_index"] for doc in docs] == [0, 5, 10]
    _assert_verbatim_and_ordered(text, docs)


def test_split_documents_carries_start_index() -> None:
    text = "See the table below.\n\nSee the table below.\n\nSummary"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30, chunk_overlap=20, add_start_index=True
    )
    docs = splitter.split_documents([Document(page_content=text, metadata={"id": 1})])

    assert [doc.metadata["start_index"] for doc in docs] == [0, 22, 44]
    assert all(doc.metadata["id"] == 1 for doc in docs)


def test_jsx_splitter_start_index_follows_dynamic_separators() -> None:
    text = "const a = 1\nfunction Box() {\n  return <Box>hi</Box>\n}\nconst b = 2"
    splitter = JSFrameworkTextSplitter(
        chunk_size=40, chunk_overlap=0, add_start_index=True
    )

    docs = splitter.create_documents([text])
    assert [doc.page_content for doc in docs] == splitter.split_text(text)
    _assert_verbatim_and_ordered(text, docs)


def test_start_index_fuzz_never_moves_backwards() -> None:
    """Seeded stand-in for the report's fuzz: indexes only stay or move forward."""
    rng = random.Random(41190)
    alphabet = "abc \n."
    for _ in range(200):
        chunk_size = rng.randint(1, 12)
        chunk_overlap = rng.randint(0, chunk_size)
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 48)))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        docs = splitter.create_documents([text])
        assert [doc.page_content for doc in docs] == splitter.split_text(text)
        _assert_verbatim_and_ordered(text, docs)
