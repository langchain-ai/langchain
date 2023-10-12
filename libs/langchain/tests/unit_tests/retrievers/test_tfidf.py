import os
from datetime import datetime
from tempfile import TemporaryDirectory

import pytest

from langchain.retrievers.tfidf import TFIDFRetriever
from langchain.schema import Document


@pytest.mark.requires("sklearn")
def test_from_texts() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    tfidf_retriever = TFIDFRetriever.from_texts(texts=input_texts)
    assert len(tfidf_retriever.docs) == 3
    assert tfidf_retriever.tfidf_array.toarray().shape == (3, 5)


@pytest.mark.requires("sklearn")
def test_from_texts_with_tfidf_params() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    tfidf_retriever = TFIDFRetriever.from_texts(
        texts=input_texts, tfidf_params={"min_df": 2}
    )
    # should count only multiple words (have, pan)
    assert tfidf_retriever.tfidf_array.toarray().shape == (3, 2)


@pytest.mark.requires("sklearn")
def test_from_documents() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    tfidf_retriever = TFIDFRetriever.from_documents(documents=input_docs)
    assert len(tfidf_retriever.docs) == 3
    assert tfidf_retriever.tfidf_array.toarray().shape == (3, 5)


@pytest.mark.requires("sklearn")
def test_save_local_load_local() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    tfidf_retriever = TFIDFRetriever.from_texts(texts=input_texts)

    file_name = "tfidf_vectorizer"
    temp_timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    with TemporaryDirectory(suffix="_" + temp_timestamp + "/") as temp_folder:
        tfidf_retriever.save_local(
            folder_path=temp_folder,
            file_name=file_name,
        )
        assert os.path.exists(os.path.join(temp_folder, f"{file_name}.joblib"))
        assert os.path.exists(os.path.join(temp_folder, f"{file_name}.pkl"))

        loaded_tfidf_retriever = TFIDFRetriever.load_local(
            folder_path=temp_folder,
            file_name=file_name,
        )
    assert len(loaded_tfidf_retriever.docs) == 3
    assert loaded_tfidf_retriever.tfidf_array.toarray().shape == (3, 5)
