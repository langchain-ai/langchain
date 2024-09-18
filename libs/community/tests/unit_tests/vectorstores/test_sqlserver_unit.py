# Unit test class
import json
import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from langchain_core.documents.base import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.sqlserver import (
    DistanceStrategy,
    SQLServer_VectorStore,
)
import sqlalchemy

EMBEDDING_LENGTH = 1536
_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO = str(
    os.environ.get("TEST_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO")
)


def generalized_mock_factory() -> None:
    mocks = {
        "_create_engine": MagicMock(),
        "_prepare_json_data_type": MagicMock(),
        "_get_embedding_store": MagicMock(),
        "_create_table_if_not_exists": MagicMock(),
        "_can_connect_with_entra_id": MagicMock(),
        "_provide_token": MagicMock(return_value=True),
        "_handle_field_filter": MagicMock(),
        "_docs_from_result": MagicMock(),
        "_docs_and_scores_from_result": MagicMock(),
        "_insert_embeddings": MagicMock(),
        "delete": MagicMock(),
        "_delete_texts_by_ids": MagicMock(),
        "similarity_search": MagicMock(),
        "similarity_search_by_vector": MagicMock(),
        "similarity_search_with_score": MagicMock(),
        "similarity_search_by_vector_with_score": MagicMock(),
        "add_texts": MagicMock(),
        "drop": MagicMock(),
        "_create_filter_clause": MagicMock(),
        "_search_store": MagicMock(),
    }

    with ExitStack() as stack:
        for method, mock in mocks.items():
            stack.enter_context(
                patch(
                    f"langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.{method}",
                    mock,
                )
            )

        connection_string = _ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO
        db_schema = "test_schema"
        distance_strategy = DistanceStrategy.DOT
        embedding_function = FakeEmbeddings(size=128)
        embedding_length = 128
        table_name = "test_table"

        store = SQLServer_VectorStore(
            connection_string=connection_string,
            db_schema=db_schema,
            distance_strategy=distance_strategy,
            embedding_function=embedding_function,
            embedding_length=embedding_length,
            table_name=table_name,
        )

    return store, mocks


def test_init():
    # Arrange
    store, mocks = generalized_mock_factory()

    # Assert
    assert store.connection_string == _ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO
    assert store._distance_strategy == DistanceStrategy.DOT
    assert store.embedding_function == FakeEmbeddings(size=128)
    assert store._embedding_length == 128
    assert store.schema == "test_schema"
    assert store.table_name == "test_table"
    mocks["_create_engine"].assert_called_once()
    mocks["_prepare_json_data_type"].assert_called_once()
    mocks["_get_embedding_store"].assert_called_once_with("test_table", "test_schema")
    mocks["_create_table_if_not_exists"].assert_called_once()


def test_can_connect_with_entra_id() -> None:
    store, mocks = generalized_mock_factory()
    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._can_connect_with_entra_id",
        wraps=SQLServer_VectorStore._can_connect_with_entra_id,
    ), patch(
        "langchain_community.vectorstores.sqlserver.urlparse", wraps=MagicMock()
    ) as mock_urlparse:
        # case 1: parsed_url is None
        mock_urlparse.return_value = None
        result = store._can_connect_with_entra_id(store)
        mock_urlparse.assert_called_once_with(store.connection_string)
        assert result is False

        mock_urlparse.reset_mock()

        # case 2: parsed_url has username and password
        url_value = {
            "username": "username123",
            "password": "password123",
        }

        json_string = json.dumps(url_value, indent=4)

        parsed_json = json.loads(
            json_string, object_hook=lambda d: SimpleNamespace(**d)
        )

        mock_urlparse.return_value = parsed_json

        result = store._can_connect_with_entra_id(store)
        mock_urlparse.assert_called_once_with(store.connection_string)
        assert result is False

        mock_urlparse.reset_mock()

        # case 3: parsed_url has trusted_connection=yes
        url_value = {
            "username": None,
            "password": None,
            "query": "trusted_connection=yes",
        }

        json_string = json.dumps(url_value, indent=4)

        parsed_json = json.loads(
            json_string, object_hook=lambda d: SimpleNamespace(**d)
        )
        mock_urlparse.return_value = parsed_json

        result = store._can_connect_with_entra_id(store)
        mock_urlparse.assert_called_once_with(store.connection_string)
        assert result is False

        mock_urlparse.reset_mock()

        # case 4: parsed_url does not have trusted_connection=yes,
        #  no username and password
        url_value = {
            "username": None,
            "password": None,
            "query": "trusted_connection=no",
        }

        json_string = json.dumps(url_value, indent=4)

        parsed_json = json.loads(
            json_string, object_hook=lambda d: SimpleNamespace(**d)
        )
        mock_urlparse.return_value = parsed_json

        result = store._can_connect_with_entra_id(store)
        mock_urlparse.assert_called_once_with(store.connection_string)
        assert result is True


def test_create_engine() -> None:
    # Arrange
    store, mocks = generalized_mock_factory()
    mocks["_can_connect_with_entra_id"].return_value = True

    # Unpatch _create_engine to call the actual method
    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine",
        wraps=SQLServer_VectorStore._create_engine,
    ), patch.object(
        store, "_can_connect_with_entra_id", wraps=mocks["_can_connect_with_entra_id"]
    ), patch.object(sqlalchemy, "create_engine", wraps=MagicMock), patch.object(
        store, "_provide_token", wraps=mocks["_provide_token"]
    ), patch("sqlalchemy.event.listen") as mock_listen:
        engine = store._create_engine(store)

    mocks["_can_connect_with_entra_id"].assert_called_once()

    if mocks["_can_connect_with_entra_id"].return_value:
        specific_calls = [
            call for call in mock_listen.call_args_list if call.args[1] == "do_connect"
        ]
        assert len(specific_calls) == 1, f"""Expected 'do_connect' to be called once.
          Called {len(specific_calls)} times."""
        specific_calls[0].assert_called_once_with(
            engine, "do_connect", store._provide_token, once=True
        )

    mock_listen.reset_mock()
    mocks["_can_connect_with_entra_id"].reset_mock()

    mocks["_can_connect_with_entra_id"].return_value = False
    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine",
        wraps=SQLServer_VectorStore._create_engine,
    ), patch.object(
        store, "_can_connect_with_entra_id", wraps=mocks["_can_connect_with_entra_id"]
    ), patch.object(store, "_provide_token", wraps=mocks["_provide_token"]), patch(
        "sqlalchemy.event.listen"
    ) as mock_listen:
        engine = store._create_engine(store)

    mocks["_can_connect_with_entra_id"].assert_called_once()

    specific_calls = [
        call for call in mock_listen.call_args_list if call.args[1] == "do_connect"
    ]
    assert (
        len(specific_calls) == 0
    ), f"Expected 'do_connect' to be called once. Called {len(specific_calls)} times."


def test_similarity_search() -> None:
    store, mocks = generalized_mock_factory()

    query = "hi"
    mock_responses = {"hello": [0.1, 0.2, 0.3], "hi": [0.01, 0.02, 0.03]}

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search",
        wraps=SQLServer_VectorStore.similarity_search,
    ), patch.object(
        store, "similarity_search_by_vector", wraps=mocks["similarity_search_by_vector"]
    ):
        store.embedding_function = Mock()
        store.embedding_function.embed_query = Mock()
        store.embedding_function.embed_query.return_value = mock_responses[query]
        store.similarity_search_by_vector.return_value = mock_responses

        store.similarity_search(store, query)

        store.similarity_search_by_vector.assert_called_once_with([0.01, 0.02, 0.03], 4)

        store.similarity_search_by_vector.reset_mock()

        query = "hello"
        store.embedding_function.embed_query.reset_mock()
        store.embedding_function.embed_query.return_value = mock_responses[query]

        store.similarity_search(store, query, 7)

        store.similarity_search_by_vector.assert_called_once_with([0.1, 0.2, 0.3], 7)


def test_similarity_search_by_vector() -> None:
    store, mocks = generalized_mock_factory()

    with (
        patch(
            "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search_by_vector",
            wraps=SQLServer_VectorStore.similarity_search_by_vector,
        ),
        patch.object(
            store,
            "similarity_search_by_vector_with_score",
            wraps=mocks["similarity_search_by_vector_with_score"],
        ),
        patch.object(store, "_docs_from_result", wraps=mocks["_docs_from_result"]),
    ):
        embeddings = [0.1, 0.2, 0.3]
        mock_responses = {
            tuple(["0.01", "0.02", "0.03"]): (
                Document(
                    page_content="""Got these on sale for roughly 25 cents per cup"""
                ),
                0.9588668232580106,
            ),
        }
        expected_result = (
            Document(page_content="Got these on sale for roughly 25 cents per cup"),
            0.9588668232580106,
        )

        store.similarity_search_by_vector_with_score.return_value = mock_responses[
            tuple(["0.01", "0.02", "0.03"])
        ]
        store._docs_from_result.return_value = expected_result

        store.similarity_search_by_vector(store, embeddings)

        store.similarity_search_by_vector_with_score.assert_called_once_with(
            embeddings, 4
        )
        store._docs_from_result.assert_called_once_with(
            mock_responses[tuple(["0.01", "0.02", "0.03"])]
        )

        store.similarity_search_by_vector_with_score.reset_mock()
        store._docs_from_result.reset_mock()

        store.similarity_search_by_vector(store, embeddings, 7)

        store._docs_from_result.assert_called_once_with(
            mock_responses[tuple(["0.01", "0.02", "0.03"])]
        )
        store.similarity_search_by_vector_with_score.assert_called_once_with(
            [0.1, 0.2, 0.3], 7
        )
        store._docs_from_result.assert_called_once_with(
            mock_responses[tuple(["0.01", "0.02", "0.03"])]
        )


def test_similarity_search_wih_score() -> None:
    store, mocks = generalized_mock_factory()

    query = "hi"
    mock_responses = {"hello": [0.1, 0.2, 0.3], "hi": [0.01, 0.02, 0.03]}

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search_with_score",
        wraps=SQLServer_VectorStore.similarity_search_with_score,
    ), patch.object(
        store,
        "similarity_search_by_vector_with_score",
        wraps=mocks["similarity_search_by_vector_with_score"],
    ):
        store.embedding_function = Mock()
        store.embedding_function.embed_query = Mock()
        store.embedding_function.embed_query.return_value = mock_responses[query]
        store.similarity_search_by_vector_with_score.return_value = mock_responses

        store.similarity_search_with_score(store, query)

        store.similarity_search_by_vector_with_score.assert_called_once_with(
            [0.01, 0.02, 0.03], 4
        )

        store.similarity_search_by_vector_with_score.reset_mock()

        query = "hello"
        store.embedding_function.embed_query.reset_mock()
        store.embedding_function.embed_query.return_value = mock_responses[query]

        store.similarity_search_with_score(store, query, 7)

        store.similarity_search_by_vector_with_score.assert_called_once_with(
            [0.1, 0.2, 0.3], 7
        )


def test_similarity_search_by_vector_with_score() -> None:
    store, mocks = generalized_mock_factory()

    with (
        patch(
            "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore."
            "similarity_search_by_vector_with_score",
            wraps=SQLServer_VectorStore.similarity_search_by_vector_with_score,
        ),
        patch.object(store, "_search_store", wraps=mocks["_search_store"]),
        patch.object(
            store,
            "_docs_and_scores_from_result",
            wraps=mocks["_docs_and_scores_from_result"],
        ),
    ):
        embeddings = tuple[0.01, 0.02, 0.03]

        expected_search_result = """(<langchain_community.vectorstores.sqlserver.
           SQLServer_VectorStore._get_embedding_store.<locals>.EmbeddingStore object
             at 0x0000025EFFF84810>,
                 0.9595672912317021)"""
        expected_docs = (
            Document(page_content="""Got these on sale for roughly 25 cents per cup"""),
            0.9588668232580106,
        )

        store._search_store.return_value = expected_search_result
        store._docs_and_scores_from_result.return_value = expected_docs

        # case 1: k is not given
        result = store.similarity_search_by_vector_with_score(store, embeddings)

        store._search_store.assert_called_once_with(embeddings, 4)
        store._docs_and_scores_from_result.assert_called_once_with(
            expected_search_result
        )
        assert result == expected_docs

        store.similarity_search_by_vector_with_score.reset_mock()
        store._docs_and_scores_from_result.reset_mock()
        store._search_store.reset_mock()

        # case 2: k =7
        result = store.similarity_search_by_vector_with_score(store, embeddings, 7)

        store._search_store.assert_called_once_with(embeddings, 7)
        store._docs_and_scores_from_result.assert_called_once_with(
            expected_search_result
        )
        assert result == expected_docs
