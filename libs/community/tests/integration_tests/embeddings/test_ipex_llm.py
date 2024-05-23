"""Test IPEX LLM"""

import os

import pytest

from langchain_community.embeddings import IpexLLMBgeEmbeddings

model_ids_to_test = os.getenv("TEST_IPEXLLM_BGE_EMBEDDING_MODEL_IDS") or ""
skip_if_no_model_ids = pytest.mark.skipif(
    not model_ids_to_test,
    reason="TEST_IPEXLLM_BGE_EMBEDDING_MODEL_IDS environment variable not set.",
)
model_ids_to_test = [model_id.strip() for model_id in model_ids_to_test.split(",")]  # type: ignore

device = os.getenv("TEST_IPEXLLM_BGE_EMBEDDING_MODEL_DEVICE") or "cpu"

sentence = "IPEX-LLM is a PyTorch library for running LLM on Intel CPU and GPU (e.g., \
local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency."
query = "What is IPEX-LLM?"


@skip_if_no_model_ids
@pytest.mark.parametrize(
    "model_id",
    model_ids_to_test,
)
def test_embed_documents(model_id: str) -> None:
    """Test IpexLLMBgeEmbeddings embed_documents"""
    embedding_model = IpexLLMBgeEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    output = embedding_model.embed_documents([sentence, query])
    assert len(output) == 2


@skip_if_no_model_ids
@pytest.mark.parametrize(
    "model_id",
    model_ids_to_test,
)
def test_embed_query(model_id: str) -> None:
    """Test IpexLLMBgeEmbeddings embed_documents"""
    embedding_model = IpexLLMBgeEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    output = embedding_model.embed_query(query)
    assert isinstance(output, list)
