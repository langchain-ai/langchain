from langchain.retrievers.tfidf import TFIDFRetriever

def test_from_texts() -> None:
    input_texts = [
        "I have a pen.",
        "Do you have a pen?",
        "I have a bag."
    ]
    tfidf_retriever = TFIDFRetriever.from_texts(texts=input_texts)
    assert len(tfidf_retriever.docs) == 3
    assert tfidf_retriever.tfidf_array.toarray().shape == (3, 5)


def test_from_texts_with_tfidf_params() -> None:
    input_texts = [
        "I have a pen.",
        "Do you have a pen?",
        "I have a bag."
    ]
    tfidf_retriever = TFIDFRetriever.from_texts(
        texts=input_texts,
        tfidf_params={"min_df": 2}
    )
    # should count only multiple words (have, pan)
    assert tfidf_retriever.tfidf_array.toarray().shape == (3, 2)

   
    