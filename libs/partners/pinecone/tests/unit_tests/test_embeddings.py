from langchain_core.utils import convert_to_secret_str

from langchain_pinecone import PineconeEmbeddings

API_KEY = convert_to_secret_str("NOT_A_VALID_KEY")
MODEL_NAME = "multilingual-e5-large"


def test_default_config() -> None:
    e = PineconeEmbeddings(
        pinecone_api_key=API_KEY,  # type: ignore[call-arg]
        model=MODEL_NAME,
    )
    assert e.batch_size == 96


def test_default_config_with_api_key() -> None:
    e = PineconeEmbeddings(api_key=API_KEY, model=MODEL_NAME)
    assert e.batch_size == 96


def test_custom_config() -> None:
    e = PineconeEmbeddings(
        pinecone_api_key=API_KEY,  # type: ignore[call-arg]
        model=MODEL_NAME,
        batch_size=128,
    )
    assert e.batch_size == 128
