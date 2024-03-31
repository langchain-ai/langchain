from unittest.mock import Mock

import pytest

from langchain_ai21 import AI21SemanticTextSplitter
from tests.unit_tests.conftest import SEGMENTS

TEXT = (
    "The original full name of the franchise is Pocket Monsters (ポケットモンスター, "
    "Poketto Monsutā), which was abbreviated to "
    "Pokemon during development of the original games.\n"
    "When the franchise was released internationally, the short form of the title was "
    "used, with an acute accent (´) "
    "over the e to aid in pronunciation.\n"
    "Pokémon refers to both the franchise itself and the creatures within its "
    "fictional universe.\n"
    "As a noun, it is identical in both the singular and plural, as is every "
    "individual species name;[10] it is "
    'grammatically correct to say "one Pokémon" and "many Pokémon", as well '
    'as "one Pikachu" and "many Pikachu".\n'
    "In English, Pokémon may be pronounced either /'powkɛmon/ (poe-keh-mon) or "
    "/'powkɪmon/ (poe-key-mon).\n"
    "The Pokémon franchise is set in a world in which humans coexist with creatures "
    "known as Pokémon.\n"
    "Pokémon Red and Blue contain 151 Pokémon species, with new ones being introduced "
    "in subsequent games; as of December 2023, 1,025 Pokémon species have been "
    "introduced.\n[b] Most Pokémon are inspired by real-world animals;[12] for example,"
    "Pikachu are a yellow mouse-like species[13] with lightning bolt-shaped tails[14] "
    "that possess electrical abilities.[15]"
)


@pytest.mark.parametrize(
    ids=[
        "when_chunk_size_is_zero",
        "when_chunk_size_is_large",
        "when_chunk_size_is_small",
    ],
    argnames=["chunk_size", "expected_segmentation_len"],
    argvalues=[
        (0, 2),
        (1000, 1),
        (10, 2),
    ],
)
def test_split_text__on_chunk_size(
    chunk_size: int,
    expected_segmentation_len: int,
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts = AI21SemanticTextSplitter(
        chunk_size=chunk_size,
        client=mock_client_with_semantic_text_splitter,
    )
    segments = sts.split_text("This is a test")
    assert len(segments) == expected_segmentation_len


def test_split_text__on_large_chunk_size__should_merge_chunks(
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts_no_merge = AI21SemanticTextSplitter(
        client=mock_client_with_semantic_text_splitter
    )
    sts_merge = AI21SemanticTextSplitter(
        client=mock_client_with_semantic_text_splitter,
        chunk_size=1000,
    )
    segments_no_merge = sts_no_merge.split_text("This is a test")
    segments_merge = sts_merge.split_text("This is a test")
    assert len(segments_merge) > 0
    assert len(segments_no_merge) > 0
    assert len(segments_no_merge) > len(segments_merge)


def test_split_text__on_small_chunk_size__should_not_merge_chunks(
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts_no_merge = AI21SemanticTextSplitter(
        client=mock_client_with_semantic_text_splitter
    )
    segments = sts_no_merge.split_text("This is a test")
    assert len(segments) == 2
    for index in range(2):
        assert segments[index] == SEGMENTS[index].segment_text


def test_create_documents__on_start_index__should_should_add_start_index(
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts = AI21SemanticTextSplitter(
        client=mock_client_with_semantic_text_splitter,
        add_start_index=True,
    )

    response = sts.create_documents(texts=[TEXT])
    assert len(response) > 0
    for segment in response:
        assert segment.page_content is not None
        assert segment.metadata is not None
        assert "start_index" in segment.metadata
        assert segment.metadata["start_index"] > -1


def test_create_documents__when_metadata_from_user__should_add_metadata(
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts = AI21SemanticTextSplitter(client=mock_client_with_semantic_text_splitter)
    metadatas = [{"hello": "world"}]
    response = sts.create_documents(texts=[TEXT], metadatas=metadatas)
    assert len(response) > 0
    for index in range(len(response)):
        assert response[index].page_content == SEGMENTS[index].segment_text
        assert len(response[index].metadata) == 2
        assert response[index].metadata["source_type"] == SEGMENTS[index].segment_type
        assert response[index].metadata["hello"] == "world"


def test_split_text_to_documents__when_metadata_not_passed__should_contain_source_type(
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts = AI21SemanticTextSplitter(client=mock_client_with_semantic_text_splitter)
    response = sts.split_text_to_documents(TEXT)
    assert len(response) > 0
    for segment in response:
        assert segment.page_content is not None
        assert segment.metadata is not None
        assert "source_type" in segment.metadata
        assert segment.metadata["source_type"] is not None
