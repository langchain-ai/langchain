import builtins
import unittest
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest

from langchain.callbacks.streamlit import StreamlitCallbackHandler


@pytest.mark.requires("streamlit")
class TestImport(unittest.TestCase):
    """Test the StreamlitCallbackHandler 'auto-updating' API"""
    def setUp(self) -> None:
        # we monkeypatch the builtins.import to control whether
        # "import streamlit.external.langchain.StreamlitCallbackHandler" is an error.
        self.python_import = builtins.__import__

    def tearDown(self) -> None:
        builtins.__import__ = self.python_import

    @mock.patch("langchain.callbacks.streamlit._InternalStreamlitCallbackHandler")
    def test_create_internal_handler(self, mock_internal_handler: Any) -> None:
        """If we're using a Streamlit that does not expose its own
        StreamlitCallbackHandler, use our own implementation.
        """

        def external_handler_import_error(
            name: str, globals: Any, locals: Any, fromlist: Any, level: int
        ) -> Any:
            if name == "streamlit.external.langchain":
                raise ImportError
            return self.python_import(name, globals, locals, fromlist, level)

        builtins.__import__ = external_handler_import_error  # type: ignore[assignment]

        parent_container = MagicMock()
        thought_labeler = MagicMock()
        StreamlitCallbackHandler(
            parent_container,
            max_thought_containers=1,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
            thought_labeler=thought_labeler,
        )

        # Our internal handler should be created
        mock_internal_handler.assert_called_once_with(
            parent_container,
            max_thought_containers=1,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
            thought_labeler=thought_labeler,
        )

    def test_create_external_handler(self) -> None:
        """If we're using a Streamlit that *does* expose its own callback handler,
        delegate to that implementation.
        """

        mock_streamlit_module = MagicMock()

        def external_handler_import_success(
            name: str, globals: Any, locals: Any, fromlist: Any, level: int
        ) -> Any:
            if name == "streamlit.external.langchain":
                return mock_streamlit_module
            return self.python_import(name, globals, locals, fromlist, level)

        builtins.__import__ = external_handler_import_success  # type: ignore[assignment] # noqa: E501

        parent_container = MagicMock()
        thought_labeler = MagicMock()
        StreamlitCallbackHandler(
            parent_container,
            max_thought_containers=1,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
            thought_labeler=thought_labeler,
        )

        # Streamlit's handler should be created
        mock_streamlit_module.StreamlitCallbackHandler.assert_called_once_with(
            parent_container,
            max_thought_containers=1,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
            thought_labeler=thought_labeler,
        )
