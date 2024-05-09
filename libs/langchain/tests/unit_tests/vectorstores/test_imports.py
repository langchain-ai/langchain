import pytest
from langchain_core.vectorstores import VectorStore

from langchain import vectorstores


@pytest.mark.community
def test_all_imports() -> None:
    """Simple test to make sure all things can be imported."""
    for cls in vectorstores.__all__:
        if cls not in [
            "AlibabaCloudOpenSearchSettings",
            "ClickhouseSettings",
            "MyScaleSettings",
        ]:
            assert issubclass(getattr(vectorstores, cls), VectorStore)
