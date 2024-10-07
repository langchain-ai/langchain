import pytest

from langchain_community.embeddings.yandex import YandexGPTEmbeddings


@pytest.mark.parametrize(
    "constructor_args",
    [
        dict(),
        dict(disable_request_logging=True),
    ],
)
# @pytest.mark.scheduled - idk what it means
# requires YC_* env and active service
def test_yandex_embedding(constructor_args: dict) -> None:
    documents = ["exactly same", "exactly same", "different"]
    embedding = YandexGPTEmbeddings(**constructor_args)
    doc_outputs = embedding.embed_documents(documents)
    assert len(doc_outputs) == 3
    for i in range(3):
        assert len(doc_outputs[i]) >= 256  # there are many dims
        assert len(doc_outputs[0]) == len(doc_outputs[i])  # dims are te same
    assert doc_outputs[0] == doc_outputs[1]  # same input, same embeddings
    assert doc_outputs[2] != doc_outputs[1]  # different input, different embeddings

    qry_output = embedding.embed_query(documents[0])
    assert len(qry_output) >= 256
    assert len(doc_outputs[0]) == len(
        qry_output
    )  # query and doc models have same dimensions
    assert doc_outputs[0] != qry_output  # query and doc models are different
