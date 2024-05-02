from typing import List

import pytest

from langchain_ai21.embeddings import _split_texts_into_batches


@pytest.mark.parametrize(
    ids=[
        "when_chunk_size_is_2__should_return_3_chunks",
        "when_texts_is_empty__should_return_empty_list",
        "when_chunk_size_is_1__should_return_10_chunks",
    ],
    argnames=["input_texts", "chunk_size", "expected_output"],
    argvalues=[
        (["a", "b", "c", "d", "e"], 2, [["a", "b"], ["c", "d"], ["e"]]),
        ([], 3, []),
        (
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            1,
            [["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"], ["10"]],
        ),
    ],
)
def test_chunked_text_generator(
    input_texts: List[str], chunk_size: int, expected_output: List[List[str]]
) -> None:
    result = list(_split_texts_into_batches(input_texts, chunk_size))
    assert result == expected_output
