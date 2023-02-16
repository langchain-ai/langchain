"""Test huggingface embeddings."""
import unittest
from typing import Any

from langchain.embeddings.self_hosted_hugging_face import (
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)


def get_remote_instance() -> Any:
    import runhouse as rh

    gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)
    gpu.install_packages(["pip:./"])
    # gpu.restart_grpc_server(resync_rh=True)
    return gpu


def test_selfhosted_huggingface_embedding_documents() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    gpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=gpu)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_selfhosted_huggingface_embedding_query() -> None:
    """Test huggingface embeddings."""
    document = "foo bar"
    gpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=gpu)
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_selfhosted_huggingface_instructor_embedding_documents() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    gpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_selfhosted_huggingface_instructor_embedding_query() -> None:
    """Test huggingface embeddings."""
    query = "foo bar"
    gpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu)
    output = embedding.embed_query(query)
    assert len(output) == 768
