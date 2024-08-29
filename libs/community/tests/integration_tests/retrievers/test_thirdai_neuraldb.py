import os
import shutil
from typing import Generator

import pytest

from langchain_community.retrievers import NeuralDBRetriever


@pytest.fixture(scope="session")
def test_csv() -> Generator[str, None, None]:
    csv = "thirdai-test.csv"
    with open(csv, "w") as o:
        o.write("column_1,column_2\n")
        o.write("column one,column two\n")
    yield csv
    os.remove(csv)


def assert_result_correctness(documents: list) -> None:
    assert len(documents) == 1
    assert documents[0].page_content == "column_1: column one\n\ncolumn_2: column two"


@pytest.mark.requires("thirdai[neural_db]")
def test_neuraldb_retriever_from_scratch(test_csv: str) -> None:
    retriever = NeuralDBRetriever.from_scratch()
    retriever.insert([test_csv])
    documents = retriever.invoke("column")
    assert_result_correctness(documents)


@pytest.mark.requires("thirdai[neural_db]")
def test_neuraldb_retriever_from_checkpoint(test_csv: str) -> None:
    checkpoint = "thirdai-test-save.ndb"
    if os.path.exists(checkpoint):
        shutil.rmtree(checkpoint)
    try:
        retriever = NeuralDBRetriever.from_scratch()
        retriever.insert([test_csv])
        retriever.save(checkpoint)
        loaded_retriever = NeuralDBRetriever.from_checkpoint(checkpoint)
        documents = loaded_retriever.invoke("column")
        assert_result_correctness(documents)
    finally:
        if os.path.exists(checkpoint):
            shutil.rmtree(checkpoint)


@pytest.mark.requires("thirdai[neural_db]")
def test_neuraldb_retriever_other_methods(test_csv: str) -> None:
    retriever = NeuralDBRetriever.from_scratch()
    retriever.insert([test_csv])
    # Make sure they don't throw an error.
    retriever.associate("A", "B")
    retriever.associate_batch([("A", "B"), ("C", "D")])
    retriever.upvote("A", 0)
    retriever.upvote_batch([("A", 0), ("B", 0)])
