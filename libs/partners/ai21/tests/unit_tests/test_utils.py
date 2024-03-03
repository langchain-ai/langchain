import pytest
from langchain_ai21.embeddings import chunked_text_generator


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
def test_chunks(input_texts, chunk_size, expected_output):
    result = list(chunked_text_generator(input_texts, chunk_size))
    assert result == expected_output
