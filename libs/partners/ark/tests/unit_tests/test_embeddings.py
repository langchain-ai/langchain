from dotenv import load_dotenv

from langchain_ark.embeddings import ArkEmbeddings

load_dotenv(override=True)


def test_ark_embeddings() -> None:
    embeddings = ArkEmbeddings()

    vector = embeddings.embed_query("Volcengine ARK")
    assert isinstance(vector, list)


def test_ark_embeddings_reduce() -> None:
    from langchain_ark.utils import cosine_similarity, sliced_norm_l2

    embeddings = ArkEmbeddings()

    vector = embeddings.embed_query("Volcengine ARK")
    dim_size = 1024
    vector_reduced = sliced_norm_l2(vector, dim_size)
    vector_a = embeddings.embed_query("Volcengine")
    vector_a_reduced = sliced_norm_l2(vector_a, dim_size)
    vector_b = embeddings.embed_query("Langchain")
    vector_b_reduced = sliced_norm_l2(vector_b, dim_size)

    assert isinstance(vector, list)
    assert isinstance(vector_reduced, list)
    assert len(vector_reduced) == dim_size

    assert cosine_similarity(vector, vector_a) > cosine_similarity(vector, vector_b)
    assert cosine_similarity(vector_reduced, vector_a_reduced) > cosine_similarity(
        vector_reduced, vector_b_reduced
    )
