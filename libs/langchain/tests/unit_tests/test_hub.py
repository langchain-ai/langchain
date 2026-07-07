import warnings
from unittest.mock import MagicMock, patch


class TestHubPullDeprecation:
    """Tests that `hub.pull` is deprecated in favor of the LangSmith SDK."""

    def test_pull_emits_deprecation(self) -> None:
        from langchain_core._api import LangChainDeprecationWarning

        from langchain_classic.hub import pull

        mock_client = MagicMock()
        mock_client.pull_prompt = MagicMock(return_value=MagicMock())

        with (
            patch("langchain_classic.hub.LangSmithClient", return_value=mock_client),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            pull("owner/repo")
            dep_warnings = [
                x for x in w if issubclass(x.category, LangChainDeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            msg = str(dep_warnings[0].message)
            assert "hub.pull" in msg
            assert "LangSmith" in msg
