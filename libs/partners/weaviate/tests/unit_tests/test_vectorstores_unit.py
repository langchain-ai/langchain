import datetime
import logging
from typing import Union

import numpy as np
import pytest

from langchain_weaviate.vectorstores import (
    WeaviateVectorStore,
    _default_score_normalizer,
    _json_serializable,
)


def test_logger_debug_level() -> None:
    """Test line 31: Verify logger is set to DEBUG level on module import."""
    import importlib
    import sys

    # Force reload to ensure line 31 is executed during this test
    if "langchain_weaviate.vectorstores" in sys.modules:
        importlib.reload(sys.modules["langchain_weaviate.vectorstores"])
    else:
        from langchain_weaviate import vectorstores  # noqa: F401

    # Get the logger from the module
    logger = logging.getLogger("langchain_weaviate.vectorstores")

    # Verify it's set to DEBUG
    assert logger.level == logging.DEBUG


@pytest.mark.parametrize("val, expected_result", [(1e6, 1.0), (-1e6, 0.0)])
def test_default_score_normalizer(val: float, expected_result: float) -> None:
    assert np.isclose(_default_score_normalizer(val), expected_result, atol=1e-6)


@pytest.mark.parametrize(
    "value, expected_result",
    [
        (datetime.datetime(2022, 1, 1, 12, 0, 0), "2022-01-01T12:00:00"),
        ("test", "test"),
        (123, 123),
        (None, None),
    ],
)
def test_json_serializable(
    value: Union[datetime.datetime, str, int, None],
    expected_result: Union[str, int, None],
) -> None:
    assert _json_serializable(value) == expected_result


def test_from_texts_raises_value_error_when_client_is_none() -> None:
    with pytest.raises(
        ValueError, match="client must be an instance of WeaviateClient"
    ):
        WeaviateVectorStore.from_texts(
            texts=["sample text"], embedding=None, client=None
        )


def test_select_relevance_score_fn_with_custom_function() -> None:
    """Test that custom relevance_score_fn is used when provided."""
    from unittest.mock import MagicMock

    def custom_score_fn(score: float) -> float:
        return score * 2.0

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = False

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="test",
        text_key="text",
        relevance_score_fn=custom_score_fn,
    )

    result_fn = vectorstore._select_relevance_score_fn()
    assert result_fn == custom_score_fn
    assert result_fn(0.5) == 1.0


def test_max_marginal_relevance_search_without_embeddings() -> None:
    """Test that MMR search raises error when no embeddings provided."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = False

    vectorstore = WeaviateVectorStore(
        client=mock_client, index_name="test", text_key="text", embedding=None
    )

    with pytest.raises(ValueError, match="requires a suitable Embeddings object"):
        vectorstore.max_marginal_relevance_search("query", k=4)


def test_vectorstore_with_custom_schema() -> None:
    """Test that WeaviateVectorStore accepts and uses custom schema."""
    from unittest.mock import MagicMock

    custom_schema = {
        "class": "CustomClass",
        "properties": [
            {"name": "custom_text", "dataType": ["text"]},
            {"name": "custom_field", "dataType": ["string"]},
        ],
        "vectorizer": "none",
    }

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = False

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="CustomClass",
        text_key="custom_text",
        schema=custom_schema,
    )

    # Verify that the custom schema was used
    assert vectorstore.schema == custom_schema
    mock_client.collections.create_from_dict.assert_called_once_with(custom_schema)


def test_vectorstore_with_default_schema() -> None:
    """Test that WeaviateVectorStore uses default schema when none provided."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = False

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="TestClass",
        text_key="text",
    )

    # Verify that default schema was created
    assert vectorstore.schema["class"] == "TestClass"
    assert vectorstore.schema["properties"][0]["name"] == "text"
    assert vectorstore.schema["MultiTenancyConfig"]["enabled"] is False
    mock_client.collections.create_from_dict.assert_called_once()


def test_vectorstore_with_custom_schema_and_multi_tenancy() -> None:
    """Test that custom schema is used as-is, even with use_multi_tenancy=True."""
    from unittest.mock import MagicMock

    custom_schema = {
        "class": "CustomClass",
        "properties": [{"name": "text", "dataType": ["text"]}],
        "MultiTenancyConfig": {"enabled": True},
    }

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = True

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="CustomClass",
        text_key="text",
        schema=custom_schema,
        use_multi_tenancy=True,
    )

    # When custom schema is provided, it should be used as-is
    # use_multi_tenancy parameter should not modify it
    assert vectorstore.schema == custom_schema
    assert vectorstore.schema["MultiTenancyConfig"]["enabled"] is True


def test_vectorstore_with_multi_tenancy_dict() -> None:
    """Test that use_multi_tenancy accepts a dict with custom config."""
    from unittest.mock import MagicMock

    multi_tenancy_config = {
        "enabled": True,
        "autoTenantCreation": False,
        "autoTenantActivation": False,
    }

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = True

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="TestClass",
        text_key="text",
        use_multi_tenancy=multi_tenancy_config,
    )

    # Should use the provided dict directly
    assert vectorstore.schema["MultiTenancyConfig"] == multi_tenancy_config
    assert vectorstore.schema["MultiTenancyConfig"]["enabled"] is True
    assert vectorstore.schema["MultiTenancyConfig"]["autoTenantCreation"] is False
    assert vectorstore.schema["MultiTenancyConfig"]["autoTenantActivation"] is False


def test_vectorstore_with_multi_tenancy_bool_true() -> None:
    """Test that use_multi_tenancy=True creates default config."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = True

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="TestClass",
        text_key="text",
        use_multi_tenancy=True,
    )

    # Should create default config with enabled=True
    assert vectorstore.schema["MultiTenancyConfig"]["enabled"] is True


def test_vectorstore_with_multi_tenancy_bool_false() -> None:
    """Test that use_multi_tenancy=False creates disabled config."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = False

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="TestClass",
        text_key="text",
        use_multi_tenancy=False,
    )

    # Should create config with multi-tenancy disabled
    assert vectorstore.schema["MultiTenancyConfig"]["enabled"] is False


def test_add_texts_raises_on_failed_objects() -> None:
    """Test that add_texts raises ValueError when batch objects fail."""
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = False

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="TestClass",
        text_key="text",
    )

    failed_obj = MagicMock()
    failed_obj.original_uuid = "uuid-1"
    failed_obj.message = "some error"

    mock_client.batch.dynamic.return_value.__enter__ = MagicMock()
    mock_client.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.batch.failed_objects = [failed_obj]

    with patch("weaviate.util.get_valid_uuid", return_value="uuid-1"):
        with pytest.raises(ValueError, match="1 object\\(s\\) failed to be added"):
            vectorstore.add_texts(["hello"])


def test_add_texts_returns_ids_on_success() -> None:
    """Test that add_texts returns all IDs when no failures occur."""
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_config = mock_client.collections.get.return_value.config.get.return_value
    mock_config.multi_tenancy_config.enabled = False

    vectorstore = WeaviateVectorStore(
        client=mock_client,
        index_name="TestClass",
        text_key="text",
    )

    mock_client.batch.dynamic.return_value.__enter__ = MagicMock()
    mock_client.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.batch.failed_objects = []

    with patch(
        "weaviate.util.get_valid_uuid",
        side_effect=["uuid-1", "uuid-2"],
    ):
        ids = vectorstore.add_texts(["hello", "world"])

    assert ids == ["uuid-1", "uuid-2"]
