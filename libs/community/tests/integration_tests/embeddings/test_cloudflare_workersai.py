"""Test Cloudflare Workers AI embeddings."""

import responses

from langchain_community.embeddings.cloudflare_workersai import (
    CloudflareWorkersAIEmbeddings,
)


@responses.activate
def test_cloudflare_workers_ai_embedding_documents() -> None:
    """Test Cloudflare Workers AI embeddings."""
    documents = ["foo bar", "foo bar", "foo bar"]

    responses.add(
        responses.POST,
        "https://api.cloudflare.com/client/v4/accounts/123/ai/run/@cf/baai/bge-base-en-v1.5",
        json={
            "result": {
                "shape": [3, 768],
                "data": [[0.0] * 768, [0.0] * 768, [0.0] * 768],
            },
            "success": "true",
            "errors": [],
            "messages": [],
        },
    )

    embeddings = CloudflareWorkersAIEmbeddings(account_id="123", api_token="abc")
    output = embeddings.embed_documents(documents)

    assert len(output) == 3
    assert len(output[0]) == 768


@responses.activate
def test_cloudflare_workers_ai_embedding_query() -> None:
    """Test Cloudflare Workers AI embeddings."""

    responses.add(
        responses.POST,
        "https://api.cloudflare.com/client/v4/accounts/123/ai/run/@cf/baai/bge-base-en-v1.5",
        json={
            "result": {"shape": [1, 768], "data": [[0.0] * 768]},
            "success": "true",
            "errors": [],
            "messages": [],
        },
    )

    document = "foo bar"
    embeddings = CloudflareWorkersAIEmbeddings(account_id="123", api_token="abc")
    output = embeddings.embed_query(document)

    assert len(output) == 768
