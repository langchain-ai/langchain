import numpy as np
import pytest

from langchain_community.embeddings.infinity_local import InfinityEmbeddingsLocal


@pytest.mark.requires("infinity_emb")
@pytest.mark.requires("torch")
@pytest.mark.asyncio
async def test_local_infinity_embeddings():
    embedder = InfinityEmbeddingsLocal(
        model="TaylorAI/bge-micro-v2",
        device="cpu",
        backend="torch",
        revision=None,
        batch_size=2,
        model_warmup=False,
    )

    async with embedder:
        embeddings = await embedder.aembed_documents(["text1", "text2", "text1"])
        assert len(embeddings) == 3
        # model has 384 dim output
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
        assert len(embeddings[2]) == 384
        # assert all different embeddings
        assert (np.array(embeddings[0]) - np.array(embeddings[1]) != 0).all()
        # assert identical embeddings, up to floating point error
        np.testing.assert_array_equal(embeddings[0], embeddings[2])


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_local_infinity_embeddings())
