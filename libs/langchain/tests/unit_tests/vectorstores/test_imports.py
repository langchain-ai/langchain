from langchain import vectorstores
from langchain.schema.vectorstore import VectorStore


def test_all_imports() -> None:
    """Simple test to make sure all things can be imported."""
    for cls in vectorstores.__all__:
        if cls not in [
            "AlibabaCloudOpenSearchSettings",
            "ClickhouseSettings",
            "MyScaleSettings",
        ]:
            assert issubclass(getattr(vectorstores, cls), VectorStore)
