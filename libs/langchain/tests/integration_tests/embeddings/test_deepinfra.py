"""Test DeepInfra API wrapper."""

from langchain.embeddings import DeepInfraEmbeddings


def test_deepinfra_call() -> None:
    """Test valid call to DeepInfra."""
    deepinfra_emb = DeepInfraEmbeddings(model_id="sentence-transformers/clip-ViT-B-32")
    r1 = deepinfra_emb.embed_documents(
        [
            "Alpha is the first letter of Greek alphabet",
            "Beta is the second letter of Greek alphabet",
        ]
    )
    assert len(r1) == 2
    assert len(r1[0]) == 512
    assert len(r1[1]) == 512
    r2 = deepinfra_emb.embed_query("What is the third letter of Greek alphabet")
    assert len(r2) == 512
