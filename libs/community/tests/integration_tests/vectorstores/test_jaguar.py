import json

from langchain_community.vectorstores.jaguar import Jaguar
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)

#############################################################################################
##
##  Requirement: fwww http server must be running at 127.0.0.1:8080 (or any end point)
##               jaguardb server must be running accepting commands from the http server
##
##  FakeEmbeddings is used to create text embeddings with dimension of 10.
##
#############################################################################################


class TestJaguar:
    vectorstore: Jaguar
    pod: str
    store: str

    @classmethod
    def setup_class(cls) -> None:
        url = "http://127.0.0.1:8080/fwww/"
        cls.pod = "vdb"
        cls.store = "langchain_test_store"
        vector_index = "v"
        vector_type = "cosine_fraction_float"
        vector_dimension = 10
        embeddings = ConsistentFakeEmbeddings()
        cls.vectorstore = Jaguar(
            cls.pod,
            cls.store,
            vector_index,
            vector_type,
            vector_dimension,
            url,
            embeddings,
        )

    @classmethod
    def teardown_class(cls) -> None:
        pass

    def test_login(self) -> None:
        """
        Requires environment variable JAGUAR_API_KEY
        or $HOME/.jagrc storing the jaguar api key
        """
        self.vectorstore.login()

    def test_create(self) -> None:
        """
        Create a vector with vector index 'v' of dimension 10
        and 'v:text' to hold text and metadatas author and category
        """
        metadata_str = "author char(32), category char(16)"
        self.vectorstore.create(metadata_str, 1024)

        podstore = self.pod + "." + self.store
        js = self.vectorstore.run(f"desc {podstore}")
        jd = json.loads(js[0])
        assert podstore in jd["data"]

    def test_add_texts(self) -> None:
        """
        Add some texts
        """
        texts = ["foo", "bar", "baz"]
        metadatas = [
            {"author": "Adam", "category": "Music"},
            {"author": "Eve", "category": "Music"},
            {"author": "John", "category": "History"},
        ]

        ids = self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        assert len(ids) == len(texts)

    def test_search(self) -> None:
        """
        Test that `foo` is closest to `foo`
        Here k is 1
        """
        output = self.vectorstore.similarity_search(
            query="foo",
            k=1,
            metadatas=["author", "category"],
        )
        assert output[0].page_content == "foo"
        assert output[0].metadata["author"] == "Adam"
        assert output[0].metadata["category"] == "Music"
        assert len(output) == 1

    def test_search_filter(self) -> None:
        """
        Test filter(where)
        """
        where = "author='Eve'"
        output = self.vectorstore.similarity_search(
            query="foo",
            k=3,
            fetch_k=9,
            where=where,
            metadatas=["author", "category"],
        )
        assert output[0].page_content == "bar"
        assert output[0].metadata["author"] == "Eve"
        assert output[0].metadata["category"] == "Music"
        assert len(output) == 1

    def test_search_anomalous(self) -> None:
        """
        Test detection of anomalousness
        """
        result = self.vectorstore.is_anomalous(
            query="dogs can jump high",
        )
        assert result is False

    def test_clear(self) -> None:
        """
        Test cleanup of data in the store
        """
        self.vectorstore.clear()
        assert self.vectorstore.count() == 0

    def test_drop(self) -> None:
        """
        Destroy the vector store
        """
        self.vectorstore.drop()

    def test_logout(self) -> None:
        """
        Logout and free resources
        """
        self.vectorstore.logout()
