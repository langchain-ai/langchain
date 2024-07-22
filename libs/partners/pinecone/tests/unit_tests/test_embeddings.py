from langchain_pinecone import PineconeEmbeddings

API_KEY = "NOT_A_VALID_KEY"
MODEL_NAME = "multilingual-e5-large"


def test_default_config():
    e = PineconeEmbeddings(pinecone_api_key=API_KEY, model=MODEL_NAME)
    assert e.batch_size == 96


def test_custom_config():
    e = PineconeEmbeddings(pinecone_api_key=API_KEY, model=MODEL_NAME, batch_size=128)
    assert e.batch_size == 128


def test_custom_model():
    e = PineconeEmbeddings(pinecone_api_key=API_KEY, model=MODEL_NAME)
    assert e.model == "a-model"
    assert e.batch_size is None
