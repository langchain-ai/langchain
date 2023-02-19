"""Test self-hosted embeddings."""
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.embeddings import (
    SelfHostedEmbeddings,
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)


def get_remote_instance() -> Any:
    """Get remote instance for testing."""
    import runhouse as rh

    gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)
    gpu.install_packages(["pip:./"])
    return gpu


def test_self_hosted_huggingface_embedding_documents() -> None:
    """Test self-hosted huggingface embeddings."""
    documents = ["foo bar"]
    gpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=gpu)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_self_hosted_huggingface_embedding_query() -> None:
    """Test self-hosted huggingface embeddings."""
    document = "foo bar"
    gpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=gpu)
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_self_hosted_huggingface_instructor_embedding_documents() -> None:
    """Test self-hosted huggingface instruct embeddings."""
    documents = ["foo bar"]
    gpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_self_hosted_huggingface_instructor_embedding_query() -> None:
    """Test self-hosted huggingface instruct embeddings."""
    query = "foo bar"
    gpu = get_remote_instance()
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu)
    output = embedding.embed_query(query)
    assert len(output) == 768


def get_pipeline() -> Any:
    """Get pipeline for testing."""
    model_id = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)


def inference_fn(pipeline: Any, prompt: str) -> Any:
    """Inference function for testing."""
    # Return last hidden state of the model
    if isinstance(prompt, list):
        return [emb[0][-1] for emb in pipeline(prompt)]
    return pipeline(prompt)[0][-1]


def test_self_hosted_embedding_documents() -> None:
    """Test self-hosted huggingface instruct embeddings."""
    documents = ["foo bar"] * 2
    gpu = get_remote_instance()
    embedding = SelfHostedEmbeddings(
        model_load_fn=get_pipeline, hardware=gpu, inference_fn=inference_fn
    )
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 50265


def test_self_hosted_embedding_query() -> None:
    """Test self-hosted custom embeddings."""
    query = "foo bar"
    gpu = get_remote_instance()
    embedding = SelfHostedEmbeddings(
        model_load_fn=get_pipeline, hardware=gpu, inference_fn=inference_fn
    )
    output = embedding.embed_query(query)
    assert len(output) == 50265
