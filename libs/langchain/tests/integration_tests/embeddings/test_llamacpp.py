# flake8: noqa
"""Test llamacpp embeddings."""
import os
from urllib.request import urlretrieve

from langchain.embeddings.llamacpp import LlamaCppEmbeddings


def get_model() -> str:
    """Download model.
    From https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/,
    convert to new ggml format and return model path.
    """
    model_url = "https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/resolve/main/ggml-alpaca-7b-q4.bin"
    tokenizer_url = "https://huggingface.co/decapoda-research/llama-7b-hf/resolve/main/tokenizer.model"
    conversion_script = "https://github.com/ggerganov/llama.cpp/raw/master/convert-unversioned-ggml-to-ggml.py"
    local_filename = model_url.split("/")[-1]

    if not os.path.exists("convert-unversioned-ggml-to-ggml.py"):
        urlretrieve(conversion_script, "convert-unversioned-ggml-to-ggml.py")
    if not os.path.exists("tokenizer.model"):
        urlretrieve(tokenizer_url, "tokenizer.model")
    if not os.path.exists(local_filename):
        urlretrieve(model_url, local_filename)
        os.system("python convert-unversioned-ggml-to-ggml.py . tokenizer.model")

    return local_filename


def test_llamacpp_embedding_documents() -> None:
    """Test llamacpp embeddings."""
    documents = ["foo bar"]
    model_path = get_model()
    embedding = LlamaCppEmbeddings(model_path=model_path)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 512


def test_llamacpp_embedding_query() -> None:
    """Test llamacpp embeddings."""
    document = "foo bar"
    model_path = get_model()
    embedding = LlamaCppEmbeddings(model_path=model_path)
    output = embedding.embed_query(document)
    assert len(output) == 512
