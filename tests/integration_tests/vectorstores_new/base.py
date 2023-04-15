# tests/integration_tests/vectorstores_new/basic.py
import logging
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import List, Type, Union

import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import VectorStore

# TODO: Move the logging setup and set the logging level in the fixture setup
# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

DEFAULT_COLLECTION_NAME = "langchain-test-collection"


class BaseTest:
    """
    Define an abstract base class that provides default implementations for
    setup and teardown methods that are run before and after the tests are run

    Attributes:
        vector_store_class: The concrete class that implements the `VectorStore`
            interface.
    """

    vector_store_class: Type[VectorStore]
    vector_store: Union[VectorStore, None] = None
    collection_name: str = DEFAULT_COLLECTION_NAME
    docsearch: Union[VectorStore, None] = None
    logger: logging.Logger

    @pytest.fixture(autouse=True)
    def _setup_logger(self) -> None:
        # Set up logger for each test class
        self.logger = logging.getLogger(self.__class__.__name__)
        # TODO: Move the logging setup and set the logging level in the fixture setup
        # Set up logging level
        logging.getLogger(__name__).setLevel(logging.DEBUG)


class FileSystemTest:
    patcher: Union[Patcher, None] = None
    tmp_directory: Union[PurePath, None] = None
    db_dir: Union[PurePath, None] = None

    @classmethod
    def setup_class(cls) -> None:
        assert cls.tmp_directory is None
        assert cls.db_dir is None
        assert cls.patcher is None

        cls.tmp_directory = PurePath(tempfile.mkdtemp())
        cls.db_dir = PurePath(os.path.join(cls.tmp_directory, tempfile.mkdtemp()))

        assert os.path.exists(cls.tmp_directory.__str__())
        assert os.path.exists(cls.db_dir.__str__())

    @classmethod
    def teardown_class(cls) -> None:
        assert cls.tmp_directory is not None
        if os.path.exists(cls.tmp_directory.__str__()):
            shutil.rmtree(cls.tmp_directory.__str__())

    def setup_method(self) -> None:
        assert self.db_dir is not None
        if os.path.exists(self.db_dir.__str__()):
            shutil.rmtree(self.db_dir.__str__())
        os.mkdir(self.db_dir.__str__())

    def teardown_method(self) -> None:
        assert self.db_dir is not None
        if os.path.exists(self.db_dir.__str__()):
            shutil.rmtree(self.db_dir.__str__())


class MixinStaticTest(BaseTest, ABC):
    """
    Mixin class containing shared test methods for vector stores.
    """

    @pytest.mark.vcr()
    def ensure_functional(self, query: str, k: int = 1) -> None:
        assert self.docsearch is not None

        output = self.docsearch.similarity_search(query, k=k)
        assert len(output) >= 1
        assert query in output[0].page_content

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def ensure_functional_async(self, query: str, k: int = 1) -> None:
        assert self.docsearch is not None

        output = await self.docsearch.asimilarity_search(query, k=k)
        assert len(output) >= 1
        assert query in output[0].page_content

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_from_texts(
        self, texts: List[str], embedding: Embeddings, query: str
    ) -> None:
        """
        Test creating a VectorStore from a list of texts.
        """
        self.logger.debug("test_from_texts")

        self.docsearch = self.vector_store_class.from_texts(
            texts=texts,
            embedding=embedding,
            collection_name=self.collection_name,
        )

        self.ensure_functional(query=query)

        await self.ensure_functional_async(query=query)

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_from_texts_with_ids(
        self, texts: List[str], embedding: Embeddings, query: str
    ) -> None:
        """
        Test adding documents to a VectorStore with ids.
        """
        ids = [uuid.uuid4().hex for _ in range(len(texts))]
        self.docsearch = self.vector_store_class.from_texts(
            texts=texts,
            embedding=embedding,
            collection_name=self.collection_name,
        )

        first = self.docsearch.similarity_search(query, k=1)

        assert len(first) >= 1
        self.docsearch.add_texts(texts, ids=ids, embedding=embedding)
        second = self.docsearch.similarity_search(query, k=1)

        #  Ensure that the texts are inserted with their respective IDs,
        #  and that no duplicates are inserted.
        assert len(first) == len(second)

        # Ensure that the texts are inserted with their respective IDs,
        # and that duplicates are not inserted because their ids are already in use.
        assert len(first) == len(second)
        ids = [uuid.uuid4().hex for _ in range(len(texts))]
        self.docsearch.add_texts(texts, ids=ids, embedding=embedding)
        duplicated = self.docsearch.similarity_search(query, k=2)

        # Ensure that at least one duplicate is inserted.
        assert len(duplicated) == len(first) * 2
        assert first[0].page_content == duplicated[0].page_content
        assert first[0].page_content == duplicated[1].page_content

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_from_texts_async(
        self, texts: List[str], embedding: Embeddings, query: str
    ) -> None:
        """
        Test creating a VectorStore from a list of texts.
        """
        self.logger.debug("test_from_texts")

        self.docsearch = await self.vector_store_class.afrom_texts(
            texts=texts,
            embedding=embedding,
            collection_name=self.collection_name,
        )

        await self.ensure_functional_async(query=query)

    @pytest.mark.vcr()
    def test_from_documents(
        self,
        documents: List[Document],
        embedding: Embeddings,
        query: str,
    ) -> None:
        """
        Test creating a VectorStore from a list of Documents.
        """
        self.logger.debug("test_from_documents")

        self.docsearch = self.vector_store_class.from_documents(
            documents=documents,
            embedding=embedding,
            collection_name=self.collection_name,
        )

        self.ensure_functional(query=query)

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_from_documents_async(
        self,
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str,
        query: str,
    ) -> None:
        """
        Test creating a VectorStore from a list of Documents.
        """
        self.logger.debug("test_from_documents")

        self.docsearch = await self.vector_store_class.afrom_documents(
            documents=documents,
            embedding=embedding,
            collection_name=collection_name,
        )

        await self.ensure_functional_async(query=query)


class StaticTest(MixinStaticTest, ABC):
    """
    Base class for remote vector store tests, which inherit the test cases
    defined in VectorStoreTestMixin.
    """

    @classmethod
    @abstractmethod
    def setup_class(cls) -> None:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def teardown_class(cls) -> None:
        raise NotImplementedError()

    @abstractmethod
    def setup_method(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def teardown_method(self) -> None:
        raise NotImplementedError()


class FilesystemTestStatic(MixinStaticTest, FileSystemTest):
    """
    Base class for remote vector store tests, which inherit the test cases
    defined in VectorStoreTestMixin.
    """


class MixinInstanceTest(BaseTest, ABC):
    embedding: Union[Embeddings, None] = None

    @classmethod
    def setup_class(cls) -> None:
        cls.embedding = OpenAIEmbeddings()

    @classmethod
    def teardown_class(cls) -> None:
        pass

    @abstractmethod
    def setup_method(self) -> None:
        raise NotImplementedError()

    @pytest.mark.vcr()
    def tests_add_texts(self, texts: List[str], query: str) -> None:
        """
        Test adding documents to a VectorStore.
        """
        self.logger.debug("test_add_texts")
        assert self.vector_store is not None

        ids_inserted = self.vector_store.add_texts(texts)
        assert len(ids_inserted) == len(texts)

        output = self.vector_store.similarity_search(query=query, k=1)
        assert len(output) >= 1

    @pytest.mark.vcr()
    def tests_add_texts_with_ids(self, texts: List[str], query: str) -> None:
        """
        Test adding documents to a VectorStore with ids.
        """
        self.logger.debug("test_add_texts")
        assert self.vector_store is not None

        ids = [uuid.uuid4().hex for _ in range(len(texts))]

        ids_inserted = self.vector_store.add_texts(texts=texts, ids=ids)
        assert len(ids_inserted) == len(texts)
        assert sorted(ids_inserted) == sorted(ids)

        output = self.vector_store.similarity_search(query=query, k=1)
        assert len(output) >= 1

    @pytest.mark.vcr()
    def tests_add_documents(self, documents: List[Document], query: str) -> None:
        """
        Test adding documents to a VectorStore.
        """
        self.logger.debug("test_add_texts")
        assert self.vector_store is not None

        self.vector_store.add_documents(documents)
        output = self.vector_store.similarity_search(query=query, k=1)
        print(output[0].page_content)
        assert len(output) >= 1


class InstanceTest(MixinInstanceTest, ABC):
    """
    Base class for remote vector store tests, which inherit the test cases
    defined in VectorStoreTestMixin.
    """


class FilesystemTestInstance(FileSystemTest, MixinInstanceTest, ABC):
    """
    Base class for remote vector store tests, which inherit the test cases
    defined in VectorStoreTestInstanceMixin.
    """

    @classmethod
    def setup_class(cls) -> None:
        # ordering is important here
        FileSystemTest.setup_class()
        MixinInstanceTest.setup_class()

    @classmethod
    def teardown_class(cls) -> None:
        # ordering is important here
        MixinInstanceTest.teardown_class()
        FileSystemTest.teardown_class()

    def setup_method(self) -> None:
        FileSystemTest.setup_method(self)
