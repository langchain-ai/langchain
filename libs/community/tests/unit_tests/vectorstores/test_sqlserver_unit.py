# Unit test class
import json
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import sqlalchemy
from langchain_core.documents.base import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.sqlserver import (
    DistanceStrategy,
    SQLServer_VectorStore,
)

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
    with (
        patch(
            "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine",
            wraps=SQLServer_VectorStore._create_engine,
        ),
        patch.object(
            store,
            "_can_connect_with_entra_id",
            wraps=mocks["_can_connect_with_entra_id"],
        ),
        patch.object(store, "_provide_token", wraps=mocks["_provide_token"]),
        patch("sqlalchemy.event.listen") as mock_listen,
        patch.object(
            sqlalchemy, "create_engine", wraps=MagicMock()
        ) as mock_create_engine,
    ):
        engine = store._create_engine(store)

    mocks["_can_connect_with_entra_id"].assert_called_once()
    mock_create_engine.return_value = MagicMock()
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
    with (
        patch(
            "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine",
            wraps=SQLServer_VectorStore._create_engine,
        ),
        patch.object(
            store,
            "_can_connect_with_entra_id",
            wraps=mocks["_can_connect_with_entra_id"],
        ),
        patch.object(
            sqlalchemy, "create_engine", wraps=MagicMock
        ) as mock_create_engine,
        patch.object(store, "_provide_token", wraps=mocks["_provide_token"]),
        patch("sqlalchemy.event.listen") as mock_listen,
    ):
        engine = store._create_engine(store)

    mock_create_engine.return_value = MagicMock()
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


def test_add_texts():
    store, mocks = generalized_mock_factory()

    texts = ["hi", "hello", "welcome"]
    embeddings = [0.01, 0.02, 0.03]
    metadatas = [
        {"id": 1, "summary": "Good Quality Dog Food"},
        {"id": 2, "summary": "Nasty No flavor"},
        {"id": 3, "summary": "stale product"},
    ]
    ids = [1, 2, 3]
    input_ids = [4, 5, 6]

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.add_texts",
        wraps=SQLServer_VectorStore.add_texts,
    ), patch.object(store, "_insert_embeddings", wraps=mocks["_insert_embeddings"]):
        store.embedding_function = Mock()
        store.embedding_function.embed_documents = Mock()
        store.embedding_function.embed_documents.return_value = embeddings
        store._insert_embeddings.return_value = ids

        # case 1:input ids not given
        returned_ids = store.add_texts(store, texts, metadatas)
        assert returned_ids == ids
        store._insert_embeddings.assert_called_once_with(
            texts, embeddings, metadatas, None
        )

        store._insert_embeddings.reset_mock()

        # case 1:input ids not given
        returned_ids = store.add_texts(store, texts, metadatas, input_ids)
        assert returned_ids == ids
        store._insert_embeddings.assert_called_once_with(
            texts, embeddings, metadatas, input_ids
        )


def test_create_filter_clause():
    store, mocks = generalized_mock_factory()
    with (
        patch(
            "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_filter_clause",
            wraps=SQLServer_VectorStore._create_filter_clause,
        ) as mock_create_filter_clause,
        patch.object(
            store, "_handle_field_filter", wraps=mocks["_handle_field_filter"]
        ),
        patch.object(sqlalchemy, "and_", wraps=MagicMock) as mock_sqlalchemy_and,
        patch.object(sqlalchemy, "or_", wraps=MagicMock) as mock_sqlalachemy_or,
    ):
        # filter case 0: Filters is not dict
        filter_value = ["hi"]

        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value)
        assert str(context.exception) == (
            """Expected a dict, but got <class 'list'> for value: <class 'filter'>"""
        )

        # filter case 1: Outer operator is not AND/OR
        filter_value = {"$XOR": 2}

        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value)
        assert str(context.exception) == (
            """Invalid filter condition.\nExpected $and or $or but got: $XOR"""
        )

        # filter case 2: Valid field filter case
        filter_value = {"id": 1}
        expected_filter_clause = """JSON_VALUE(langchain_vector_store_tests.
        content_metadata, :JSON_VALUE_1) = :JSON_VALUE_2"""
        store._handle_field_filter.return_value = expected_filter_clause

        filter_clause_returned = store._create_filter_clause(store, filter_value)
        assert filter_clause_returned == expected_filter_clause
        store._handle_field_filter.assert_called_once_with("id", 1)
        store._handle_field_filter.reset_mock()

        # filter case 3 - Filter value is not list
        filter_value = {"$or": {"hi"}}

        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value)
        assert (
            str(context.exception)
            == """Expected a list, but got <class 'set'> for value: {'hi'}"""
        )
        store._handle_field_filter.reset_mock()

        # filter case 4 - length of fields >1 and have operator, not fields
        filter_value = {"$eq": {}, "$gte": 1}
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value)
        assert (
            str(context.exception)
            == """Invalid filter condition. Expected a field but got: $eq"""
        )
        store._handle_field_filter.reset_mock()

        # filter case 5 - length of fields > 1 and have all fields, we AND it together

        filter_value = {
            "id": {"$eq": [1, 5, 2, 9]},
            "location": {"$eq": ["pond", "market"]},
        }
        expected_filter_clause = (
            "JSON_VALUE(langchain_vector_store_tests.content_metadata, :JSON_VALUE_1)"
            " = :JSON_VALUE_2 AND "
            "JSON_VALUE(langchain_vector_store_tests.content_metadata,"
            " :JSON_VALUE_3) = :JSON_VALUE_4"
        )
        mock_sqlalchemy_and.return_value = expected_filter_clause
        store._create_filter_clause(store, filter_value)
        assert store._handle_field_filter.call_count == 2
        store._handle_field_filter.reset_mock()

        # filter case 6 - empty dictionary
        filter_value = {}
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value)
        assert str(context.exception) == """Got an empty dictionary for filters."""

        # filter case 7 - filter is None
        filter_value = None
        filter_clause_returned = store._create_filter_clause(store, filter_value)
        assert filter_clause_returned is None


def test_handle_field_filter():
    store, mocks = generalized_mock_factory()

    with (
        patch(
            "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._handle_field_filter",
            wraps=SQLServer_VectorStore._handle_field_filter,
        ),
        patch.object(sqlalchemy, "and_", wraps=MagicMock) as mock_sqlalchemy_and,
        patch.object(sqlalchemy, "or_", wraps=MagicMock),
        patch.object(
            sqlalchemy.sql.operators, "ne", wraps=MagicMock
        ) as mock_sqlalchemy_ne,
        patch.object(
            sqlalchemy.sql.operators, "lt", wraps=MagicMock
        ) as mock_sqlalchemy_lt,
        patch.object(
            sqlalchemy.sql.operators, "ge", wraps=MagicMock
        ) as mock_sqlalchemy_gte,
        patch.object(
            sqlalchemy.sql.operators, "le", wraps=MagicMock
        ) as mock_sqlalchemy_lte,
    ):
        # Test case 1: field startWith $
        field = "$AND"
        value = 1
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._handle_field_filter(store, field, value)
        assert (
            str(context.exception)
            == "Invalid filter condition. Expected a field but got an operator: $AND"
        )

        # Test case 2: field is not valid identifier
        field = "/?"
        value = 1
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._handle_field_filter(store, field, value)
        assert (
            str(context.exception)
            == f"Invalid field name: {field}. Expected a valid identifier."
        )

        # Test case 3: more than 1 filter for value
        field = "id"
        value = {"id": "3", "name": "john"}
        expected_message = (
            "Invalid filter condition."
            " Expected a value which is a dictionary with a single key "
            "that corresponds to an operator but got a dictionary with 2 keys."
            " The first few keys are: "
            "['id', 'name']"
        )
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._handle_field_filter(store, field, value)

        assert str(context.exception) == expected_message

        # Test case 4: field is not valid identifier
        field = "id"
        value = {"$neee": 1}
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._handle_field_filter(store, field, value)
        assert str(context.exception).startswith("Invalid operator: $neee.")

        # Test case 5: SPECIAL CASED OPERATORS
        field = "id"
        value = {"$ne": 1}
        expected_response = """JSON_VALUE(langchain_vector_store_tests.content_metadata,
          :JSON_VALUE_1) != :JSON_VALUE_2"""
        mock_sqlalchemy_ne.return_value = expected_response
        handle_field_filter_response = store._handle_field_filter(store, field, value)
        assert handle_field_filter_response == expected_response

        # Test case 6: NUMERIC OPERATORS
        field = "id"
        value = {"$lt": 1}
        expected_response = """JSON_VALUE(langchain_vector_store_tests.content_metadata,
          :JSON_VALUE_1) < :JSON_VALUE_2"""
        mock_sqlalchemy_lt.return_value = expected_response
        handle_field_filter_response = store._handle_field_filter(store, field, value)
        assert handle_field_filter_response == expected_response

        # Test case 7: BETWEEN OPERATOR
        field = "id"
        value = {"$between": (1, 2)}
        expected_response = """CAST(JSON_VALUE(
        langchain_vector_store_tests.content_metadata, :JSON_VALUE_1) AS
         NUMERIC(10, 2)) >= :param_1 AND 
         CAST(JSON_VALUE(langchain_vector_store_tests.content_metadata,
          :JSON_VALUE_1) AS NUMERIC(10, 2)) <= :param_2"""
        mock_sqlalchemy_lte.return_value = (
            "CAST(JSON_VALUE(langchain_vector_store_tests.content_metadata,"
            " :JSON_VALUE_1) AS NUMERIC(10, 2)) <= :param_2"
        )
        mock_sqlalchemy_gte.return_value = (
            "CAST(JSON_VALUE(langchain_vector_store_tests.content_metadata,"
            " :JSON_VALUE_1) AS NUMERIC(10, 2)) >= :param_1 "
        )
        mock_sqlalchemy_and.return_value = expected_response
        handle_field_filter_response = store._handle_field_filter(store, field, value)
        assert handle_field_filter_response == expected_response
        mock_sqlalchemy_and.assert_called_once_with(
            mock_sqlalchemy_gte.return_value, mock_sqlalchemy_lte.return_value
        )

        # Test case 8: SPECIAL CASED OPERATOR unsupported
        field = "id"
        value = {"$in": [[], []]}
        with unittest.TestCase().assertRaises(NotImplementedError) as context:
            store._handle_field_filter(store, field, value)
        assert (
            str(context.exception) == "Unsupported type: <class 'list'> for value: []"
        )

        # Test case 9: SPECIAL CASED OPERATOR IN
        field = "id"
        value = {"$in": ["adam", "bob"]}
        expected_response = (
            "JSON_VALUE(:JSON_VALUE_1, :JSON_VALUE_2) IN (__[POSTCOMPILE_JSON_VALUE_3])"
        )
        handle_field_filter_response = store._handle_field_filter(store, field, value)
        assert str(handle_field_filter_response) == expected_response

        # Test case 10: SPECIAL CASED OPERATOR LIKE
        field = "id"
        value = {"$like": ["adam", "bob"]}
        expected_response = (
            "JSON_VALUE(:JSON_VALUE_1, :JSON_VALUE_2) LIKE :JSON_VALUE_3"
        )
        handle_field_filter_response = store._handle_field_filter(store, field, value)
        assert str(handle_field_filter_response) == expected_response


def test_docs_from_result():
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._docs_from_result",
        wraps=SQLServer_VectorStore._docs_from_result,
    ):
        result = [
            (
                Document(
                    page_content="id 3",
                    metadata={
                        "name": "jane",
                        "date": "2021-01-01",
                        "count": 3,
                        "is_active": True,
                        "tags": ["b", "d"],
                        "location": [3.0, 4.0],
                        "id": 3,
                        "height": 2.4,
                        "happiness": None,
                    },
                ),
                0.982679262929245,
            ),
            (
                Document(
                    page_content="id 1",
                    metadata={
                        "name": "adam",
                        "date": "2021-01-01",
                        "count": 1,
                        "is_active": True,
                        "tags": ["a", "b"],
                        "location": [1.0, 2.0],
                        "id": 1,
                        "height": 10.0,
                        "happiness": 0.9,
                        "sadness": 0.1,
                    },
                ),
                1.0078365850902349,
            ),
        ]
        expected_documents = [
            Document(
                page_content="id 3",
                metadata={
                    "name": "jane",
                    "date": "2021-01-01",
                    "count": 3,
                    "is_active": True,
                    "tags": ["b", "d"],
                    "location": [3.0, 4.0],
                    "id": 3,
                    "height": 2.4,
                    "happiness": None,
                },
            ),
            Document(
                page_content="id 1",
                metadata={
                    "name": "adam",
                    "date": "2021-01-01",
                    "count": 1,
                    "is_active": True,
                    "tags": ["a", "b"],
                    "location": [1.0, 2.0],
                    "id": 1,
                    "height": 10.0,
                    "happiness": 0.9,
                    "sadness": 0.1,
                },
            ),
        ]
        documents_returned = store._docs_from_result(store, result)

        assert documents_returned == expected_documents


def test_docs_and_scores_from_result():
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._docs_and_scores_from_result",
        wraps=SQLServer_VectorStore._docs_and_scores_from_result,
    ):
        result = [
            SimpleNamespace(
                EmbeddingStore=SimpleNamespace(
                    content="hi", content_metadata={"key": "value"}
                ),
                distance=1,
            )
        ]

        expected_documents = [
            (Document(page_content="hi", metadata={"key": "value"}), 1)
        ]
        resulted_docs_and_score = store._docs_and_scores_from_result(store, result)

        assert resulted_docs_and_score == expected_documents


def test_delete():
    store, mocks = generalized_mock_factory()

    with (
        patch(
            "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.delete",
            wraps=SQLServer_VectorStore.delete,
        ),
        patch.object(
            store, "_delete_texts_by_ids", wraps=mocks["_delete_texts_by_ids"]
        ),
    ):
        ids = None
        assert store.delete(store, ids) is False

        ids = []
        assert store.delete(store, ids) is False

        ids = [1, 2, 3]
        store._delete_texts_by_ids.return_value = 0
        assert store.delete(store, ids) is False

        ids = [1, 2, 3]
        store._delete_texts_by_ids.return_value = 1
        assert store.delete(store, ids) is True
