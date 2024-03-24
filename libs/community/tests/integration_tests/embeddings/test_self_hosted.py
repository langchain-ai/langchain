"""Test self-hosted embeddings."""
from typing import Any

from langchain_community.embeddings import (
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)


def get_remote_instance() -> Any:
    """Get remote instance for testing."""
    import runhouse as rh

    gpu = rh.cluster(name="rh-a10x", instance_type="g5.4xlarge", use_spot=False)
    gpu.run(commands=["pip install langchain"])
    return gpu


def get_remote_env(gpu: Any) -> Any:
    import runhouse as rh

    embedding_env = rh.env(
        name="embeddings_env",
        reqs=[
            "transformers",
            "torch",
            "accelerate",
            "huggingface-hub",
            "sentence_transformers",
        ],
        secrets=["huggingface"],  # need for downloading models from huggingface
    ).to(system=gpu)

    return embedding_env


def test_self_hosted_huggingface_embedding_documents() -> None:
    """Test self-hosted huggingface embeddings."""
    documents = ["foo bar"]
    gpu = get_remote_instance()
    env = get_remote_env(gpu)
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=gpu, env=env)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_self_hosted_huggingface_embedding_query() -> None:
    """Test self-hosted huggingface embeddings."""
    document = "foo bar"
    gpu = get_remote_instance()
    env = get_remote_env(gpu)
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=gpu, env=env)
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_self_hosted_huggingface_instructor_embedding_documents() -> None:
    """Test self-hosted huggingface instruct embeddings."""
    documents = ["foo bar"]
    gpu = get_remote_instance()
    env = get_remote_env(gpu)
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu, env=env)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_self_hosted_huggingface_instructor_embedding_query() -> None:
    """Test self-hosted huggingface instruct embeddings."""
    query = "foo bar"
    gpu = get_remote_instance()
    env = get_remote_env(gpu)
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu, env=env)
    output = embedding.embed_query(query)
    assert len(output) == 768
