import pytest
from langchain_chroma.vectorstores import Chroma
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

class TestChromaReadWriteTestSuite(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> Chroma:
        store = Chroma(embedding_function=self.get_embeddings())
        store.reset_collection()
        yield store
        

class TestSync(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> Chroma:
        store = Chroma(embedding_function=self.get_embeddings())
        store.reset_collection()
        yield store