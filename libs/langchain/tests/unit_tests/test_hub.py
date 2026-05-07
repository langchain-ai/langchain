import warnings
from unittest.mock import MagicMock, patch


class TestHubDeprecation:
    """Tests `hub` is deprecated in favor of the LangSmith SDK."""

    def test_push_emits_deprecation(self) -> None:
        from langchain_core._api.deprecation import (
            LangChainPendingDeprecationWarning,
        )

        from langchain.hub import push

        mock_client = MagicMock()
        mock_client.push_prompt = MagicMock(return_value="https://example.com")

        with (
            patch("langchain.hub.LangSmithClient", return_value=mock_client),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            push("owner/repo", object=MagicMock())
            dep_warnings = [
                x
                for x in w
                if issubclass(x.category, LangChainPendingDeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            msg = str(dep_warnings[0].message)
            assert "hub.push" in msg
            assert "LangSmith" in msg

    def test_pull_emits_deprecation(self) -> None:
        from langchain_core._api.deprecation import (
            LangChainPendingDeprecationWarning,
        )

        from langchain.hub import pull

        mock_client = MagicMock()
        mock_client.pull_prompt = MagicMock(return_value=MagicMock())

        with (
            patch("langchain.hub.LangSmithClient", return_value=mock_client),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            pull("owner/repo")
            dep_warnings = [
                x
                for x in w
                if issubclass(x.category, LangChainPendingDeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            msg = str(dep_warnings[0].message)
            assert "hub.pull" in msg
            assert "LangSmith" in msg
