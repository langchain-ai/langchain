import unittest

from langchain.document_loaders.fauna import FaunaLoader

try:
    import fauna  # noqa: F401

    fauna_installed = True
except ImportError:
    fauna_installed = False


@unittest.skipIf(not fauna_installed, "fauna not installed")
class TestFaunaLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.fauna_secret = "<enter-valid-fauna-secret>"
        self.valid_fql_query = "Item.all()"
        self.valid_page_content_field = "text"
        self.valid_metadata_fields = ["valid_metadata_fields"]

    def test_fauna_loader(self) -> None:
        """Test Fauna loader."""
        loader = FaunaLoader(
            query=self.valid_fql_query,
            page_content_field=self.valid_page_content_field,
            secret=self.fauna_secret,
            metadata_fields=self.valid_metadata_fields,
        )
        docs = loader.load()

        assert len(docs) > 0  # assuming the query returns at least one document
        for doc in docs:
            assert (
                doc.page_content != ""
            )  # assuming that every document has page_content
            assert (
                "id" in doc.metadata and doc.metadata["id"] != ""
            )  # assuming that every document has 'id'
            assert (
                "ts" in doc.metadata and doc.metadata["ts"] != ""
            )  # assuming that every document has 'ts'
