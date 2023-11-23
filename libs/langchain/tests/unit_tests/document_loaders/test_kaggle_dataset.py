import os

import pandas as pd
import pytest
from pytest import fixture

from langchain.document_loaders.kaggle_dataset import KaggleDatasetLoader

TEST_DATASET_PATH = "./kaggle_dataset.csv"


@fixture(autouse=True)
def setup_and_tear_down() -> None:
    if os.path.isfile(TEST_DATASET_PATH):
        return True
    data = {"col1": [1, 2], "col2": [3, 4], "col3": ["5", "6"]}
    pd.DataFrame(data=data).to_csv(TEST_DATASET_PATH, index=False)
    yield
    if os.path.isfile(TEST_DATASET_PATH):
        os.remove(TEST_DATASET_PATH)


def test_raise_error_if_path_not_exist() -> None:
    assert os.path.isfile(TEST_DATASET_PATH)
    os.remove(TEST_DATASET_PATH)
    with pytest.raises(FileNotFoundError):
        loader = KaggleDatasetLoader(
            dataset_path=TEST_DATASET_PATH, page_content_column="col3"
        )
        loader.load()


def test_raise_error_if_wrong_() -> None:
    with pytest.raises(AssertionError):
        loader = KaggleDatasetLoader(
            dataset_path=TEST_DATASET_PATH, page_content_column="wrong_column"
        )
        loader.load()


def test_success() -> None:
    loader = KaggleDatasetLoader(
        dataset_path=TEST_DATASET_PATH, page_content_column="col3"
    )
    docs = loader.load()
    assert len(docs) == 2
    assert docs[0].page_content == "5"
    assert docs[1].page_content == "6"
    assert docs[0].metadata == {"col1": 1, "col2": 3}
    assert docs[1].metadata == {"col1": 2, "col2": 4}
