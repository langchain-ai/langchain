"""Unit tests for SuperlinkedRetriever."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from langchain_superlinked.retrievers import SuperlinkedRetriever


class TestSuperlinkedRetrieverCore:
    """Test core functionality by directly testing the _get_relevant_documents method."""

    def test_get_relevant_documents_success(self) -> None:
        """Test successful document retrieval."""
        # Create a minimal mock retriever by patching just what we need
        retriever = Mock(spec=SuperlinkedRetriever)

        # Mock the attributes
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = None
        retriever.k = 4  # Add k parameter

        # Mock the query result
        mock_entry1 = Mock()
        mock_entry1.id = "1"
        mock_entry1.fields = {"text": "The Eiffel Tower is in Paris.", "category": "landmark"}

        mock_entry2 = Mock()
        mock_entry2.id = "2"
        mock_entry2.fields = {"text": "The Colosseum is in Rome.", "category": "landmark"}

        mock_result = Mock()
        mock_result.entries = [mock_entry1, mock_entry2]
        retriever.sl_client.query.return_value = mock_result

        # Test the actual method implementation
        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        docs = SuperlinkedRetriever._get_relevant_documents(retriever, "landmarks", run_manager=run_manager)

        # Verify results
        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "The Eiffel Tower is in Paris."
        assert docs[0].metadata == {"id": "1", "category": "landmark"}
        assert docs[1].page_content == "The Colosseum is in Rome."
        assert docs[1].metadata == {"id": "2", "category": "landmark"}

    def test_get_relevant_documents_with_k_parameter(self) -> None:
        """Test retrieval with k parameter limiting results."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = None
        retriever.k = 4  # Default k

        # Create 5 mock entries
        mock_entries = []
        for i in range(5):
            mock_entry = Mock()
            mock_entry.id = str(i)
            mock_entry.fields = {"text": f"Document {i}"}
            mock_entries.append(mock_entry)

        mock_result = Mock()
        mock_result.entries = mock_entries
        retriever.sl_client.query.return_value = mock_result

        # Test with k=2 override
        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        docs = SuperlinkedRetriever._get_relevant_documents(
            retriever,
            "test",
            run_manager=run_manager,
            k=2  # Override k parameter
        )

        # Should return only 2 documents
        assert len(docs) == 2
        assert docs[0].page_content == "Document 0"
        assert docs[1].page_content == "Document 1"

    def test_get_relevant_documents_with_specific_metadata_fields(self) -> None:
        """Test retrieval with specific metadata fields configuration."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = ["category", "score"]
        retriever.k = 4

        mock_entry = Mock()
        mock_entry.id = "1"
        mock_entry.fields = {
            "text": "Test content",
            "category": "test",
            "score": 0.95,
            "unwanted_field": "should_not_appear"
        }

        mock_result = Mock()
        mock_result.entries = [mock_entry]
        retriever.sl_client.query.return_value = mock_result

        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        docs = SuperlinkedRetriever._get_relevant_documents(retriever, "test", run_manager=run_manager)

        # Verify only specified metadata fields are included
        assert len(docs) == 1
        assert docs[0].metadata == {"id": "1", "category": "test", "score": 0.95}
        assert "unwanted_field" not in docs[0].metadata

    def test_get_relevant_documents_missing_page_content_field(self) -> None:
        """Test behavior when page_content_field is missing from results."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = None
        retriever.k = 4

        mock_entry = Mock()
        mock_entry.id = "1"
        mock_entry.fields = {"category": "test"}  # Missing "text" field

        mock_result = Mock()
        mock_result.entries = [mock_entry]
        retriever.sl_client.query.return_value = mock_result

        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        docs = SuperlinkedRetriever._get_relevant_documents(retriever, "test", run_manager=run_manager)

        # Entry should be skipped when page_content_field is missing
        assert len(docs) == 0

    def test_get_relevant_documents_empty_fields(self) -> None:
        """Test behavior with entries that have None or empty fields."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = None
        retriever.k = 4

        mock_entry = Mock()
        mock_entry.id = "1"
        mock_entry.fields = None

        mock_result = Mock()
        mock_result.entries = [mock_entry]
        retriever.sl_client.query.return_value = mock_result

        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        docs = SuperlinkedRetriever._get_relevant_documents(retriever, "test", run_manager=run_manager)

        # Entry should be skipped when fields is None
        assert len(docs) == 0

    def test_get_relevant_documents_query_exception(self) -> None:
        """Test error handling when Superlinked query raises an exception."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = None
        retriever.k = 4

        retriever.sl_client.query.side_effect = Exception("Superlinked query failed")

        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()

        # Should capture stdout to verify error message is printed
        with patch("builtins.print") as mock_print:
            docs = SuperlinkedRetriever._get_relevant_documents(retriever, "test", run_manager=run_manager)

            # Should return empty list on error
            assert docs == []

            # Should print error message
            mock_print.assert_called_once_with("Error executing Superlinked query: Superlinked query failed")

    def test_get_relevant_documents_empty_results(self) -> None:
        """Test behavior with empty query results."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = None
        retriever.k = 4

        mock_result = Mock()
        mock_result.entries = []
        retriever.sl_client.query.return_value = mock_result

        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        docs = SuperlinkedRetriever._get_relevant_documents(retriever, "test", run_manager=run_manager)

        assert docs == []

    def test_metadata_includes_id(self) -> None:
        """Test that document metadata always includes the entry ID."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = None
        retriever.k = 4

        mock_entry = Mock()
        mock_entry.id = "test_id_123"
        mock_entry.fields = {"text": "Test content"}

        mock_result = Mock()
        mock_result.entries = [mock_entry]
        retriever.sl_client.query.return_value = mock_result

        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        docs = SuperlinkedRetriever._get_relevant_documents(retriever, "test", run_manager=run_manager)

        assert len(docs) == 1
        assert docs[0].metadata["id"] == "test_id_123"

    def test_custom_query_parameter(self) -> None:
        """Test retrieval with custom query text parameter."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "search_term"  # Custom parameter name
        retriever.metadata_fields = None
        retriever.k = 4

        mock_result = Mock()
        mock_result.entries = []
        retriever.sl_client.query.return_value = mock_result

        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        SuperlinkedRetriever._get_relevant_documents(retriever, "test query", run_manager=run_manager)

        # Verify custom parameter name was used
        retriever.sl_client.query.assert_called_once_with(
            query_descriptor=retriever.sl_query,
            search_term="test query"
        )

    def test_additional_kwargs(self) -> None:
        """Test retrieval with additional query parameters."""
        retriever = Mock(spec=SuperlinkedRetriever)
        retriever.sl_client = Mock()
        retriever.sl_query = Mock()
        retriever.page_content_field = "text"
        retriever.query_text_param = "query_text"
        retriever.metadata_fields = None
        retriever.k = 4

        mock_result = Mock()
        mock_result.entries = []
        retriever.sl_client.query.return_value = mock_result

        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        SuperlinkedRetriever._get_relevant_documents(
            retriever,
            "test query",
            run_manager=run_manager,
            limit=5,
            filter_param="value"
        )

        # Verify additional parameters were passed
        retriever.sl_client.query.assert_called_once_with(
            query_descriptor=retriever.sl_query,
            query_text="test query",
            limit=5,
            filter_param="value"
        )


class TestSuperlinkedRetrieverClassProperties:
    """Test class properties and configuration without instantiation."""

    def test_class_inheritance(self) -> None:
        """Test that SuperlinkedRetriever properly inherits from BaseRetriever."""
        from langchain_core.retrievers import BaseRetriever
        assert issubclass(SuperlinkedRetriever, BaseRetriever)

    def test_config_properties(self) -> None:
        """Test that the retriever has proper pydantic configuration."""
        assert hasattr(SuperlinkedRetriever, 'Config')
        assert SuperlinkedRetriever.Config.arbitrary_types_allowed is True

    def test_field_definitions(self) -> None:
        """Test that required fields are defined."""
        # Handle both Pydantic V1 and V2
        try:
            fields = SuperlinkedRetriever.model_fields  # Pydantic V2
        except AttributeError:
            fields = SuperlinkedRetriever.__fields__  # Pydantic V1

        # Check that all required fields exist
        required_fields = ["sl_client", "sl_query", "page_content_field", "query_text_param", "metadata_fields", "k"]
        for field_name in required_fields:
            assert field_name in fields, f"Field {field_name} not found"

        # Test that query_text_param field exists and is properly configured
        query_text_field = fields["query_text_param"]
        assert query_text_field is not None

        # Verify it has the expected default by checking the string representation
        field_str = str(query_text_field)
        assert "default='query_text'" in field_str, f"Field definition should contain default='query_text', got: {field_str}"

        # Test that k field has correct default
        k_field = fields["k"]
        k_field_str = str(k_field)
        assert "default=4" in k_field_str, f"k field should have default=4, got: {k_field_str}"

    def test_field_descriptions_exist(self) -> None:
        """Test that field descriptions are defined."""
        # Handle both Pydantic V1 and V2
        try:
            fields = SuperlinkedRetriever.model_fields  # Pydantic V2
        except AttributeError:
            fields = SuperlinkedRetriever.__fields__  # Pydantic V1

        # Check that key fields have descriptions
        assert hasattr(fields["sl_client"], 'description')
        assert hasattr(fields["sl_query"], 'description')
        assert hasattr(fields["page_content_field"], 'description')
        assert hasattr(fields["query_text_param"], 'description')
        assert hasattr(fields["metadata_fields"], 'description')
        assert hasattr(fields["k"], 'description')


class TestSuperlinkedRetrieverValidation:
    """Test validation scenarios (mocked to avoid actual superlinked dependency)."""

    def test_validation_import_error(self) -> None:
        """Test that missing superlinked package is handled."""
        # This test verifies the validation method exists and can be called
        # In real usage, it would check for the superlinked package
        assert hasattr(SuperlinkedRetriever, 'validate_superlinked_packages')
        assert callable(SuperlinkedRetriever.validate_superlinked_packages)

    def test_validation_method_signature(self) -> None:
        """Test that validation method has correct signature."""
        import inspect
        sig = inspect.signature(SuperlinkedRetriever.validate_superlinked_packages)
        params = list(sig.parameters.keys())
        # For a @root_validator decorated classmethod, the first parameter is 'values'
        # The 'cls' parameter is implicit since it's a classmethod
        assert 'values' in params
        # Verify it's the expected root validator signature
        assert len(params) == 1  # Should only have 'values' parameter
