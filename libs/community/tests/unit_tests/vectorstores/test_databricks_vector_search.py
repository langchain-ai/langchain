import itertools
import random
import uuid
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.vectorstores import DatabricksVectorSearch
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)

DEFAULT_VECTOR_DIMENSION = 4


class FakeEmbeddingsWithDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def __init__(self, dimension: int = DEFAULT_VECTOR_DIMENSION):
        super().__init__()
        self.dimension = dimension

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (self.dimension - 1) + [float(i)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (self.dimension - 1) + [float(0.0)]


DEFAULT_EMBEDDING_MODEL = FakeEmbeddingsWithDimension()
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_VECTOR_COLUMN = "text_vector"
DEFAULT_PRIMARY_KEY = "id"

DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS = {
    "name": "ml.llm.index",
    "endpoint_name": "vector_search_endpoint",
    "index_type": "DELTA_SYNC",
    "primary_key": DEFAULT_PRIMARY_KEY,
    "delta_sync_index_spec": {
        "source_table": "ml.llm.source_table",
        "pipeline_type": "CONTINUOUS",
        "embedding_source_columns": [
            {
                "name": DEFAULT_TEXT_COLUMN,
                "embedding_model_endpoint_name": "openai-text-embedding",
            }
        ],
    },
}

DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS = {
    "name": "ml.llm.index",
    "endpoint_name": "vector_search_endpoint",
    "index_type": "DELTA_SYNC",
    "primary_key": DEFAULT_PRIMARY_KEY,
    "delta_sync_index_spec": {
        "source_table": "ml.llm.source_table",
        "pipeline_type": "CONTINUOUS",
        "embedding_vector_columns": [
            {
                "name": DEFAULT_VECTOR_COLUMN,
                "embedding_dimension": DEFAULT_VECTOR_DIMENSION,
            }
        ],
    },
}

DIRECT_ACCESS_INDEX = {
    "name": "ml.llm.index",
    "endpoint_name": "vector_search_endpoint",
    "index_type": "DIRECT_ACCESS",
    "primary_key": DEFAULT_PRIMARY_KEY,
    "direct_access_index_spec": {
        "embedding_vector_columns": [
            {
                "name": DEFAULT_VECTOR_COLUMN,
                "embedding_dimension": DEFAULT_VECTOR_DIMENSION,
            }
        ],
        "schema_json": f"{{"
        f'"{DEFAULT_PRIMARY_KEY}": "int", '
        f'"feat1": "str", '
        f'"feat2": "float", '
        f'"text": "string", '
        f'"{DEFAULT_VECTOR_COLUMN}": "array<float>"'
        f"}}",
    },
}

ALL_INDEXES = [
    DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS,
    DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS,
    DIRECT_ACCESS_INDEX,
]

EXAMPLE_SEARCH_RESPONSE = {
    "manifest": {
        "column_count": 3,
        "columns": [
            {"name": DEFAULT_PRIMARY_KEY},
            {"name": DEFAULT_TEXT_COLUMN},
            {"name": "score"},
        ],
    },
    "result": {
        "row_count": len(fake_texts),
        "data_array": sorted(
            [[str(uuid.uuid4()), s, random.uniform(0, 1)] for s in fake_texts],
            key=lambda x: x[2],  # type: ignore
            reverse=True,
        ),
    },
    "next_page_token": "",
}

EXAMPLE_SEARCH_RESPONSE_FIXED_SCORE: Dict = {
    "manifest": {
        "column_count": 3,
        "columns": [
            {"name": DEFAULT_PRIMARY_KEY},
            {"name": DEFAULT_TEXT_COLUMN},
            {"name": "score"},
        ],
    },
    "result": {
        "row_count": len(fake_texts),
        "data_array": sorted(
            [[str(uuid.uuid4()), s, 0.5] for s in fake_texts],
            key=lambda x: x[2],  # type: ignore
            reverse=True,
        ),
    },
    "next_page_token": "",
}

EXAMPLE_SEARCH_RESPONSE_WITH_EMBEDDING = {
    "manifest": {
        "column_count": 3,
        "columns": [
            {"name": DEFAULT_PRIMARY_KEY},
            {"name": DEFAULT_TEXT_COLUMN},
            {"name": DEFAULT_VECTOR_COLUMN},
            {"name": "score"},
        ],
    },
    "result": {
        "row_count": len(fake_texts),
        "data_array": sorted(
            [
                [str(uuid.uuid4()), s, e, random.uniform(0, 1)]
                for s, e in zip(
                    fake_texts, DEFAULT_EMBEDDING_MODEL.embed_documents(fake_texts)
                )
            ],
            key=lambda x: x[2],  # type: ignore
            reverse=True,
        ),
    },
    "next_page_token": "",
}

ALL_QUERY_TYPES = [
    None,
    "ANN",
    "HYBRID",
]


def mock_index(index_details: dict) -> MagicMock:
    from databricks.vector_search.client import VectorSearchIndex

    index = MagicMock(spec=VectorSearchIndex)
    index.describe.return_value = index_details
    return index


def default_databricks_vector_search(
    index: MagicMock, columns: Optional[List[str]] = None
) -> DatabricksVectorSearch:
    return DatabricksVectorSearch(
        index,
        embedding=DEFAULT_EMBEDDING_MODEL,
        text_column=DEFAULT_TEXT_COLUMN,
        columns=columns,
    )


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_init_delta_sync_with_managed_embeddings() -> None:
    index = mock_index(DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS)
    vectorsearch = DatabricksVectorSearch(index)
    assert vectorsearch.index == index


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_init_delta_sync_with_self_managed_embeddings() -> None:
    index = mock_index(DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS)
    vectorsearch = DatabricksVectorSearch(
        index,
        embedding=DEFAULT_EMBEDDING_MODEL,
        text_column=DEFAULT_TEXT_COLUMN,
    )
    assert vectorsearch.index == index


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_init_direct_access_index() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorsearch = DatabricksVectorSearch(
        index,
        embedding=DEFAULT_EMBEDDING_MODEL,
        text_column=DEFAULT_TEXT_COLUMN,
    )
    assert vectorsearch.index == index


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_init_fail_no_index() -> None:
    with pytest.raises(TypeError):
        DatabricksVectorSearch()  # type: ignore[call-arg]


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_init_fail_index_none() -> None:
    with pytest.raises(TypeError) as ex:
        DatabricksVectorSearch(None)
    assert "index must be of type VectorSearchIndex." in str(ex.value)


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_init_fail_text_column_mismatch() -> None:
    index = mock_index(DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS)
    with pytest.raises(ValueError) as ex:
        DatabricksVectorSearch(
            index,
            text_column="some_other_column",
        )
    assert (
        f"text_column 'some_other_column' does not match with the source column of the "
        f"index: '{DEFAULT_TEXT_COLUMN}'." in str(ex.value)
    )


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details", [DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS, DIRECT_ACCESS_INDEX]
)
def test_init_fail_no_text_column(index_details: dict) -> None:
    index = mock_index(index_details)
    with pytest.raises(ValueError) as ex:
        DatabricksVectorSearch(
            index,
            embedding=DEFAULT_EMBEDDING_MODEL,
        )
    assert "`text_column` is required for this index." in str(ex.value)


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize("index_details", [DIRECT_ACCESS_INDEX])
def test_init_fail_columns_not_in_schema(index_details: dict) -> None:
    index = mock_index(index_details)
    with pytest.raises(ValueError) as ex:
        DatabricksVectorSearch(
            index,
            embedding=DEFAULT_EMBEDDING_MODEL,
            text_column=DEFAULT_TEXT_COLUMN,
            columns=["some_random_column"],
        )
    assert "column 'some_random_column' is not in the index's schema." in str(ex.value)


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details", [DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS, DIRECT_ACCESS_INDEX]
)
def test_init_fail_no_embedding(index_details: dict) -> None:
    index = mock_index(index_details)
    with pytest.raises(ValueError) as ex:
        DatabricksVectorSearch(
            index,
            text_column=DEFAULT_TEXT_COLUMN,
        )
    assert "`embedding` is required for this index." in str(ex.value)


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details", [DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS, DIRECT_ACCESS_INDEX]
)
def test_init_fail_embedding_dim_mismatch(index_details: dict) -> None:
    index = mock_index(index_details)
    with pytest.raises(ValueError) as ex:
        DatabricksVectorSearch(
            index,
            text_column=DEFAULT_TEXT_COLUMN,
            embedding=FakeEmbeddingsWithDimension(DEFAULT_VECTOR_DIMENSION + 1),
        )
    assert (
        f"embedding model's dimension '{DEFAULT_VECTOR_DIMENSION + 1}' does not match "
        f"with the index's dimension '{DEFAULT_VECTOR_DIMENSION}'"
    ) in str(ex.value)


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_from_texts_not_supported() -> None:
    with pytest.raises(NotImplementedError) as ex:
        DatabricksVectorSearch.from_texts(fake_texts, FakeEmbeddings())
    assert (
        "`from_texts` is not supported. "
        "Use `add_texts` to add to existing direct-access index."
    ) in str(ex.value)


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details",
    [DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS, DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS],
)
def test_add_texts_not_supported_for_delta_sync_index(index_details: dict) -> None:
    index = mock_index(index_details)
    vectorsearch = default_databricks_vector_search(index)
    with pytest.raises(ValueError) as ex:
        vectorsearch.add_texts(fake_texts)
    assert "`add_texts` is only supported for direct-access index." in str(ex.value)


def is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_add_texts() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorsearch = DatabricksVectorSearch(
        index,
        embedding=DEFAULT_EMBEDDING_MODEL,
        text_column=DEFAULT_TEXT_COLUMN,
    )
    ids = [idx for idx, i in enumerate(fake_texts)]
    vectors = DEFAULT_EMBEDDING_MODEL.embed_documents(fake_texts)

    added_ids = vectorsearch.add_texts(fake_texts, ids=ids)
    index.upsert.assert_called_once_with(
        [
            {
                DEFAULT_PRIMARY_KEY: id_,
                DEFAULT_TEXT_COLUMN: text,
                DEFAULT_VECTOR_COLUMN: vector,
            }
            for text, vector, id_ in zip(fake_texts, vectors, ids)
        ]
    )
    assert len(added_ids) == len(fake_texts)
    assert added_ids == ids


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_add_texts_handle_single_text() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorsearch = DatabricksVectorSearch(
        index,
        embedding=DEFAULT_EMBEDDING_MODEL,
        text_column=DEFAULT_TEXT_COLUMN,
    )
    vectors = DEFAULT_EMBEDDING_MODEL.embed_documents(fake_texts)

    added_ids = vectorsearch.add_texts(fake_texts[0])
    index.upsert.assert_called_once_with(
        [
            {
                DEFAULT_PRIMARY_KEY: id_,
                DEFAULT_TEXT_COLUMN: text,
                DEFAULT_VECTOR_COLUMN: vector,
            }
            for text, vector, id_ in zip(fake_texts, vectors, added_ids)
        ]
    )
    assert len(added_ids) == 1
    assert is_valid_uuid(added_ids[0])


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_add_texts_with_default_id() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorsearch = default_databricks_vector_search(index)
    vectors = DEFAULT_EMBEDDING_MODEL.embed_documents(fake_texts)

    added_ids = vectorsearch.add_texts(fake_texts)
    index.upsert.assert_called_once_with(
        [
            {
                DEFAULT_PRIMARY_KEY: id_,
                DEFAULT_TEXT_COLUMN: text,
                DEFAULT_VECTOR_COLUMN: vector,
            }
            for text, vector, id_ in zip(fake_texts, vectors, added_ids)
        ]
    )
    assert len(added_ids) == len(fake_texts)
    assert all([is_valid_uuid(id_) for id_ in added_ids])


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_add_texts_with_metadata() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorsearch = default_databricks_vector_search(index)
    vectors = DEFAULT_EMBEDDING_MODEL.embed_documents(fake_texts)
    metadatas = [{"feat1": str(i), "feat2": i + 1000} for i in range(len(fake_texts))]

    added_ids = vectorsearch.add_texts(fake_texts, metadatas=metadatas)
    index.upsert.assert_called_once_with(
        [
            {
                DEFAULT_PRIMARY_KEY: id_,
                DEFAULT_TEXT_COLUMN: text,
                DEFAULT_VECTOR_COLUMN: vector,
                **metadata,  # type: ignore[arg-type]
            }
            for text, vector, id_, metadata in zip(
                fake_texts, vectors, added_ids, metadatas
            )
        ]
    )
    assert len(added_ids) == len(fake_texts)
    assert all([is_valid_uuid(id_) for id_ in added_ids])


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details",
    [DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS, DIRECT_ACCESS_INDEX],
)
def test_embeddings_property(index_details: dict) -> None:
    index = mock_index(index_details)
    vectorsearch = default_databricks_vector_search(index)
    assert vectorsearch.embeddings == DEFAULT_EMBEDDING_MODEL


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details",
    [DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS, DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS],
)
def test_delete_not_supported_for_delta_sync_index(index_details: dict) -> None:
    index = mock_index(index_details)
    vectorsearch = default_databricks_vector_search(index)
    with pytest.raises(ValueError) as ex:
        vectorsearch.delete(["some id"])
    assert "`delete` is only supported for direct-access index." in str(ex.value)


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_delete() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorsearch = default_databricks_vector_search(index)

    vectorsearch.delete(["some id"])
    index.delete.assert_called_once_with(["some id"])


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_delete_fail_no_ids() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorsearch = default_databricks_vector_search(index)

    with pytest.raises(ValueError) as ex:
        vectorsearch.delete()
    assert "ids must be provided." in str(ex.value)


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details, query_type", itertools.product(ALL_INDEXES, ALL_QUERY_TYPES)
)
def test_similarity_search(index_details: dict, query_type: Optional[str]) -> None:
    index = mock_index(index_details)
    index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE
    vectorsearch = default_databricks_vector_search(index)
    query = "foo"
    filters = {"some filter": True}
    limit = 7

    search_result = vectorsearch.similarity_search(
        query, k=limit, filter=filters, query_type=query_type
    )
    if index_details == DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS:
        index.similarity_search.assert_called_once_with(
            columns=[DEFAULT_PRIMARY_KEY, DEFAULT_TEXT_COLUMN],
            query_text=query,
            query_vector=None,
            filters=filters,
            num_results=limit,
            query_type=query_type,
        )
    else:
        index.similarity_search.assert_called_once_with(
            columns=[DEFAULT_PRIMARY_KEY, DEFAULT_TEXT_COLUMN],
            query_text=None,
            query_vector=DEFAULT_EMBEDDING_MODEL.embed_query(query),
            filters=filters,
            num_results=limit,
            query_type=query_type,
        )
    assert len(search_result) == len(fake_texts)
    assert sorted([d.page_content for d in search_result]) == sorted(fake_texts)
    assert all([DEFAULT_PRIMARY_KEY in d.metadata for d in search_result])


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_similarity_search_both_filter_and_filters_passed() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE
    vectorsearch = default_databricks_vector_search(index)
    query = "foo"
    filter = {"some filter": True}
    filters = {"some other filter": False}

    vectorsearch.similarity_search(query, filter=filter, filters=filters)
    index.similarity_search.assert_called_once_with(
        columns=[DEFAULT_PRIMARY_KEY, DEFAULT_TEXT_COLUMN],
        query_vector=DEFAULT_EMBEDDING_MODEL.embed_query(query),
        # `filter` should prevail over `filters`
        filters=filter,
        num_results=4,
        query_text=None,
        query_type=None,
    )


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details, columns, expected_columns",
    [
        (DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS, None, {"id"}),
        (DIRECT_ACCESS_INDEX, None, {"id"}),
        (
            DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS,
            [DEFAULT_PRIMARY_KEY, DEFAULT_TEXT_COLUMN, DEFAULT_VECTOR_COLUMN],
            {"text_vector", "id"},
        ),
        (
            DIRECT_ACCESS_INDEX,
            [DEFAULT_PRIMARY_KEY, DEFAULT_TEXT_COLUMN, DEFAULT_VECTOR_COLUMN],
            {"text_vector", "id"},
        ),
    ],
)
def test_mmr_search(
    index_details: dict, columns: Optional[List[str]], expected_columns: Set[str]
) -> None:
    index = mock_index(index_details)
    index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE_WITH_EMBEDDING
    vectorsearch = default_databricks_vector_search(index, columns)
    query = fake_texts[0]
    filters = {"some filter": True}
    limit = 1

    search_result = vectorsearch.max_marginal_relevance_search(
        query, k=limit, filters=filters
    )
    assert [doc.page_content for doc in search_result] == [fake_texts[0]]
    assert [set(doc.metadata.keys()) for doc in search_result] == [expected_columns]


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details", [DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS, DIRECT_ACCESS_INDEX]
)
def test_mmr_parameters(index_details: dict) -> None:
    index = mock_index(index_details)
    index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE_WITH_EMBEDDING
    query = fake_texts[0]
    limit = 1
    fetch_k = 3
    lambda_mult = 0.25
    filters = {"some filter": True}

    with patch(
        "langchain_community.vectorstores.databricks_vector_search.maximal_marginal_relevance"
    ) as mock_mmr:
        mock_mmr.return_value = [2]
        retriever = default_databricks_vector_search(index).as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": limit,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                "filter": filters,
            },
        )
        search_result = retriever.invoke(query)

    mock_mmr.assert_called_once()
    assert mock_mmr.call_args[1]["lambda_mult"] == lambda_mult
    assert index.similarity_search.call_args[1]["num_results"] == fetch_k
    assert index.similarity_search.call_args[1]["filters"] == filters
    assert len(search_result) == limit


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details, threshold", itertools.product(ALL_INDEXES, [0.4, 0.5, 0.8])
)
def test_similarity_score_threshold(index_details: dict, threshold: float) -> None:
    index = mock_index(index_details)
    index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE_FIXED_SCORE
    uniform_response_score = EXAMPLE_SEARCH_RESPONSE_FIXED_SCORE["result"][
        "data_array"
    ][0][2]
    query = fake_texts[0]
    limit = len(fake_texts)

    retriever = default_databricks_vector_search(index).as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": limit, "score_threshold": threshold},
    )
    search_result = retriever.invoke(query)
    if uniform_response_score >= threshold:
        assert len(search_result) == len(fake_texts)
    else:
        assert len(search_result) == 0


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_standard_params() -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorstore = default_databricks_vector_search(index)
    retriever = vectorstore.as_retriever()
    ls_params = retriever._get_ls_params()
    assert ls_params == {
        "ls_retriever_name": "vectorstore",
        "ls_vector_store_provider": "DatabricksVectorSearch",
        "ls_embedding_provider": "FakeEmbeddingsWithDimension",
    }

    index = mock_index(DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS)
    vectorstore = default_databricks_vector_search(index)
    retriever = vectorstore.as_retriever()
    ls_params = retriever._get_ls_params()
    assert ls_params == {
        "ls_retriever_name": "vectorstore",
        "ls_vector_store_provider": "DatabricksVectorSearch",
    }


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "index_details", [DELTA_SYNC_INDEX_SELF_MANAGED_EMBEDDINGS, DIRECT_ACCESS_INDEX]
)
def test_similarity_search_by_vector(index_details: dict) -> None:
    index = mock_index(index_details)
    index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE
    vectorsearch = default_databricks_vector_search(index)
    query_embedding = DEFAULT_EMBEDDING_MODEL.embed_query("foo")
    filters = {"some filter": True}
    limit = 7

    search_result = vectorsearch.similarity_search_by_vector(
        query_embedding, k=limit, filter=filters
    )
    index.similarity_search.assert_called_once_with(
        columns=[DEFAULT_PRIMARY_KEY, DEFAULT_TEXT_COLUMN],
        query_vector=query_embedding,
        filters=filters,
        num_results=limit,
        query_type=None,
    )
    assert len(search_result) == len(fake_texts)
    assert sorted([d.page_content for d in search_result]) == sorted(fake_texts)
    assert all([DEFAULT_PRIMARY_KEY in d.metadata for d in search_result])


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize("index_details", ALL_INDEXES)
def test_similarity_search_empty_result(index_details: dict) -> None:
    index = mock_index(index_details)
    index.similarity_search.return_value = {
        "manifest": {
            "column_count": 3,
            "columns": [
                {"name": DEFAULT_PRIMARY_KEY},
                {"name": DEFAULT_TEXT_COLUMN},
                {"name": "score"},
            ],
        },
        "result": {
            "row_count": 0,
            "data_array": [],
        },
        "next_page_token": "",
    }
    vectorsearch = default_databricks_vector_search(index)

    search_result = vectorsearch.similarity_search("foo")
    assert len(search_result) == 0


@pytest.mark.requires("databricks", "databricks.vector_search")
def test_similarity_search_by_vector_not_supported_for_managed_embedding() -> None:
    index = mock_index(DELTA_SYNC_INDEX_MANAGED_EMBEDDINGS)
    index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE
    vectorsearch = default_databricks_vector_search(index)
    query_embedding = DEFAULT_EMBEDDING_MODEL.embed_query("foo")
    filters = {"some filter": True}
    limit = 7

    with pytest.raises(ValueError) as ex:
        vectorsearch.similarity_search_by_vector(
            query_embedding, k=limit, filters=filters
        )
    assert (
        "`similarity_search_by_vector` is not supported for index with "
        "Databricks-managed embeddings." in str(ex.value)
    )


@pytest.mark.requires("databricks", "databricks.vector_search")
@pytest.mark.parametrize(
    "method",
    [
        "similarity_search",
        "similarity_search_with_score",
        "similarity_search_by_vector",
        "similarity_search_by_vector_with_score",
        "max_marginal_relevance_search",
        "max_marginal_relevance_search_by_vector",
    ],
)
def test_filter_arg_alias(method: str) -> None:
    index = mock_index(DIRECT_ACCESS_INDEX)
    vectorsearch = default_databricks_vector_search(index)
    query = "foo"
    query_embedding = DEFAULT_EMBEDDING_MODEL.embed_query("foo")
    filters = {"some filter": True}
    limit = 7

    if "by_vector" in method:
        getattr(vectorsearch, method)(query_embedding, k=limit, filters=filters)
    else:
        getattr(vectorsearch, method)(query, k=limit, filters=filters)

    index_call_args = index.similarity_search.call_args[1]
    assert index_call_args["filters"] == filters
