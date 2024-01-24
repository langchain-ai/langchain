"""Test DeepInfra API wrapper."""

from langchain_community.embeddings import DeepInfraEmbeddings


def test_deepinfra_call() -> None:
    """Test valid call to DeepInfra."""
    deepinfra_emb = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5")
    r1 = deepinfra_emb.embed_documents(
        [
            "Alpha is the first letter of Greek alphabet",
            "Beta is the second letter of Greek alphabet",
        ]
    )
    assert len(r1) == 2
    assert len(r1[0]) == 768
    assert len(r1[1]) == 768
    r2 = deepinfra_emb.embed_query("What is the third letter of Greek alphabet")
    assert len(r2) == 768
