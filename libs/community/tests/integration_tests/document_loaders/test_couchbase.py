import unittest

from langchain_community.document_loaders.couchbase import CouchbaseLoader

try:
    import couchbase  # noqa: F401

    couchbase_installed = True
except ImportError:
    couchbase_installed = False


@unittest.skipIf(not couchbase_installed, "couchbase not installed")
class TestCouchbaseLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.conn_string = "<enter-valid-couchbase-connection-string>"
        self.database_user = "<enter-valid-couchbase-user>"
        self.database_password = "<enter-valid-couchbase-password>"
        self.valid_query = "select h.* from `travel-sample`.inventory.hotel h limit 10"
        self.valid_page_content_fields = ["country", "name", "description"]
        self.valid_metadata_fields = ["id"]

    def test_couchbase_loader(self) -> None:
        """Test Couchbase loader."""
        loader = CouchbaseLoader(
            connection_string=self.conn_string,
            db_username=self.database_user,
            db_password=self.database_password,
            query=self.valid_query,
            page_content_fields=self.valid_page_content_fields,
            metadata_fields=self.valid_metadata_fields,
        )
        docs = loader.load()
        print(docs)  # noqa: T201

        assert len(docs) > 0  # assuming the query returns at least one document
        for doc in docs:
            print(doc)  # noqa: T201
            assert (
                doc.page_content != ""
            )  # assuming that every document has page_content
            assert (
                "id" in doc.metadata and doc.metadata["id"] != ""
            )  # assuming that every document has 'id'
