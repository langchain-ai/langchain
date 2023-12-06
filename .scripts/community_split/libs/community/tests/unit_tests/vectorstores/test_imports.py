from langchain_community.vectorstores import __all__
from langchain_core.vectorstores import VectorStore


def test_all_imports() -> None:
    """Simple test to make sure all things can be imported."""
    for cls in __all__:
        if cls not in [
            "AlibabaCloudOpenSearchSettings",
            "ClickhouseSettings",
            "MyScaleSettings",
        ]:
            assert issubclass(getattr(vectorstores, cls), VectorStore)
