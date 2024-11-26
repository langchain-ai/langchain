"""Test self-hosted embeddings."""

from typing import Any

from langchain_community.embeddings import (
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)


def get_remote_instance() -> Any:
    """Get remote instance for testing using HPU."""
    import runhouse as rh

    # Intel Gaudi instance
    hpu = rh.cluster(name="gaudi-instance", instance_type="dl1.24xlarge")
    hpu.install_packages(["pip:./"])
    return hpu


def test_self_hosted_huggingface_embedding_documents_hpu() -> None:
    """Test self-hosted huggingface embeddings using HPU."""
    documents = ["foo bar"]
    hpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=hpu)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_self_hosted_huggingface_embedding_query_hpu() -> None:
    """Test self-hosted huggingface embeddings using HPU."""
    document = "foo bar"
    hpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=hpu)
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_self_hosted_huggingface_instructor_embedding_documents_hpu() -> None:
    """Test self-hosted huggingface instruct embeddings using HPU."""
    documents = ["foo bar"]
    hpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=hpu)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_self_hosted_huggingface_instructor_embedding_query_hpu() -> None:
    """Test self-hosted huggingface instruct embeddings using HPU."""
    query = "foo bar"
    hpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=hpu)
    output = embedding.embed_query(query)
    assert len(output) == 768
