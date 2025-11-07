"""Test suite to test `VectorStore` integrations."""

from abc import abstractmethod

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_tests.base import BaseStandardTests

# Arbitrarily chosen. Using a small embedding size
# so tests are faster and easier to debug.
EMBEDDING_SIZE = 6


def _sort_by_id(documents: list[Document]) -> list[Document]:
    return sorted(documents, key=lambda doc: doc.id or "")


class VectorStoreIntegrationTests(BaseStandardTests):
    """Base class for vector store integration tests.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty vector store for each test.

    The fixture should use the `get_embeddings` method to get a pre-defined
    embeddings model that should be used for this test suite.

    Here is a template:

    ```python
    from typing import Generator

    import pytest
    from langchain_core.vectorstores import VectorStore
    from langchain_parrot_link.vectorstores import ParrotVectorStore
    from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests


    class TestParrotVectorStore(VectorStoreIntegrationTests):
        @pytest.fixture()
        def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
            \"\"\"Get an empty vectorstore.\"\"\"
            store = ParrotVectorStore(self.get_embeddings())
            # note: store should be EMPTY at this point
            # if you need to delete data, you may do so here
            try:
                yield store
            finally:
                # cleanup operations, or deleting data
                pass
    ```

    In the fixture, before the `yield` we instantiate an empty vector store. In the
    `finally` block, we call whatever logic is necessary to bring the vector store
    to a clean state.

    ```python
    from typing import Generator

    import pytest
    from langchain_core.vectorstores import VectorStore
    from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

    from langchain_chroma import Chroma


    class TestChromaStandard(VectorStoreIntegrationTests):
        @pytest.fixture()
        def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
            \"\"\"Get an empty VectorStore for unit tests.\"\"\"
            store = Chroma(embedding_function=self.get_embeddings())
            try:
                yield store
            finally:
                store.delete_collection()
                pass
    ```

    Note that by default we enable both sync and async tests. To disable either,
    override the `has_sync` or `has_async` properties to `False` in the
    subclass. For example:

    ```python
    class TestParrotVectorStore(VectorStoreIntegrationTests):
        @pytest.fixture()
        def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
            ...

        @property
        def has_async(self) -> bool:
            return False
    ```

    !!! note
        API references for individual test methods include troubleshooting tips.
    """  # noqa: E501

    @abstractmethod
    @pytest.fixture
    def vectorstore(self) -> VectorStore:
        """Get the `VectorStore` class to test.

        The returned `VectorStore` should be empty.
        """

    @property
    def has_sync(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return True

    @property
    def has_async(self) -> bool:
        """Configurable property to enable or disable async tests."""
        return True

    @property
    def has_get_by_ids(self) -> bool:
        """Whether the `VectorStore` supports `get_by_ids`."""
        return True

    @staticmethod
    def get_embeddings() -> Embeddings:
        """Get embeddings.

        A pre-defined embeddings model that should be used for this test.

        This currently uses `DeterministicFakeEmbedding` from `langchain-core`,
        which uses numpy to generate random numbers based on a hash of the input text.

        The resulting embeddings are not meaningful, but they are deterministic.
        """
        return DeterministicFakeEmbedding(
            size=EMBEDDING_SIZE,
        )

    def test_vectorstore_is_empty(self, vectorstore: VectorStore) -> None:
        """Test that the `VectorStore` is empty.

        ??? note "Troubleshooting"

            If this test fails, check that the test class (i.e., sub class of
            `VectorStoreIntegrationTests`) initializes an empty vector store in the
            `vectorestore` fixture.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        assert vectorstore.similarity_search("foo", k=1) == []

    def test_add_documents(self, vectorstore: VectorStore) -> None:
        """Test adding documents into the `VectorStore`.

        ??? note "Troubleshooting"

            If this test fails, check that:

            1. We correctly initialize an empty vector store in the `vectorestore`
                fixture.
            2. Calling `similarity_search` for the top `k` similar documents does
                not threshold by score.
            3. We do not mutate the original document object when adding it to the
                vector store (e.g., by adding an ID).
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        original_documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(original_documents)
        documents = vectorstore.similarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
        ]
        # Verify that the original document object does not get mutated!
        # (e.g., an ID is added to the original document object)
        assert original_documents == [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]

    def test_vectorstore_still_empty(self, vectorstore: VectorStore) -> None:
        """Test that the `VectorStore` is still empty.

        This test should follow a test that adds documents.

        This just verifies that the fixture is set up properly to be empty
        after each test.

        ??? note "Troubleshooting"

            If this test fails, check that the test class (i.e., sub class of
            `VectorStoreIntegrationTests`) correctly clears the vector store in the
            `finally` block.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        assert vectorstore.similarity_search("foo", k=1) == []

    def test_deleting_documents(self, vectorstore: VectorStore) -> None:
        """Test deleting documents from the `VectorStore`.

        ??? note "Troubleshooting"

            If this test fails, check that `add_documents` preserves identifiers
            passed in through `ids`, and that `delete` correctly removes
            documents.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents, ids=["1", "2"])
        assert ids == ["1", "2"]
        vectorstore.delete(["1"])
        documents = vectorstore.similarity_search("foo", k=1)
        assert documents == [Document(page_content="bar", metadata={"id": 2}, id="2")]

    def test_deleting_bulk_documents(self, vectorstore: VectorStore) -> None:
        """Test that we can delete several documents at once.

        ??? note "Troubleshooting"

            If this test fails, check that `delete` correctly removes multiple
            documents when given a list of IDs.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
            Document(page_content="baz", metadata={"id": 3}),
        ]

        vectorstore.add_documents(documents, ids=["1", "2", "3"])
        vectorstore.delete(["1", "2"])
        documents = vectorstore.similarity_search("foo", k=1)
        assert documents == [Document(page_content="baz", metadata={"id": 3}, id="3")]

    def test_delete_missing_content(self, vectorstore: VectorStore) -> None:
        """Deleting missing content should not raise an exception.

        ??? note "Troubleshooting"

            If this test fails, check that `delete` does not raise an exception
            when deleting IDs that do not exist.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        vectorstore.delete(["1"])
        vectorstore.delete(["1", "2", "3"])

    def test_add_documents_with_ids_is_idempotent(
        self, vectorstore: VectorStore
    ) -> None:
        """Adding by ID should be idempotent.

        ??? note "Troubleshooting"

            If this test fails, check that adding the same document twice with the
            same IDs has the same effect as adding it once (i.e., it does not
            duplicate the documents).
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        vectorstore.add_documents(documents, ids=["1", "2"])
        vectorstore.add_documents(documents, ids=["1", "2"])
        documents = vectorstore.similarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id="2"),
            Document(page_content="foo", metadata={"id": 1}, id="1"),
        ]

    def test_add_documents_by_id_with_mutation(self, vectorstore: VectorStore) -> None:
        """Test that we can overwrite by ID using `add_documents`.

        ??? note "Troubleshooting"

            If this test fails, check that when `add_documents` is called with an
            ID that already exists in the vector store, the content is updated
            rather than duplicated.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]

        vectorstore.add_documents(documents=documents, ids=["1", "2"])

        # Now over-write content of ID 1
        new_documents = [
            Document(
                page_content="new foo", metadata={"id": 1, "some_other_field": "foo"}
            ),
        ]

        vectorstore.add_documents(documents=new_documents, ids=["1"])

        # Check that the content has been updated
        documents = vectorstore.similarity_search("new foo", k=2)
        assert documents == [
            Document(
                id="1",
                page_content="new foo",
                metadata={"id": 1, "some_other_field": "foo"},
            ),
            Document(id="2", page_content="bar", metadata={"id": 2}),
        ]

    def test_get_by_ids(self, vectorstore: VectorStore) -> None:
        """Test get by IDs.

        This test requires that `get_by_ids` be implemented on the vector store.

        ??? note "Troubleshooting"

            If this test fails, check that `get_by_ids` is implemented and returns
            documents in the same order as the IDs passed in.

            !!! note
                `get_by_ids` was added to the `VectorStore` interface in
                `langchain-core` version 0.2.11. If difficult to implement, this
                test can be skipped by setting the `has_get_by_ids` property to
                `False`.

                ```python
                @property
                def has_get_by_ids(self) -> bool:
                    return False
                ```
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        if not self.has_get_by_ids:
            pytest.skip("get_by_ids not implemented.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents, ids=["1", "2"])
        retrieved_documents = vectorstore.get_by_ids(ids)
        assert _sort_by_id(retrieved_documents) == _sort_by_id(
            [
                Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
                Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            ]
        )

    def test_get_by_ids_missing(self, vectorstore: VectorStore) -> None:
        """Test get by IDs with missing IDs.

        ??? note "Troubleshooting"

            If this test fails, check that `get_by_ids` is implemented and does not
            raise an exception when given IDs that do not exist.

            !!! note
                `get_by_ids` was added to the `VectorStore` interface in
                `langchain-core` version 0.2.11. If difficult to implement, this
                test can be skipped by setting the `has_get_by_ids` property to
                `False`.

                ```python
                @property
                def has_get_by_ids(self) -> bool:
                    return False
                ```
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        if not self.has_get_by_ids:
            pytest.skip("get_by_ids not implemented.")

        # This should not raise an exception
        documents = vectorstore.get_by_ids(["1", "2", "3"])
        assert documents == []

    def test_add_documents_documents(self, vectorstore: VectorStore) -> None:
        """Run `add_documents` tests.

        ??? note "Troubleshooting"

            If this test fails, check that `get_by_ids` is implemented and returns
            documents in the same order as the IDs passed in.

            Check also that `add_documents` will correctly generate string IDs if
            none are provided.

            !!! note
                `get_by_ids` was added to the `VectorStore` interface in
                `langchain-core` version 0.2.11. If difficult to implement, this
                test can be skipped by setting the `has_get_by_ids` property to
                `False`.

                ```python
                @property
                def has_get_by_ids(self) -> bool:
                    return False
                ```
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        if not self.has_get_by_ids:
            pytest.skip("get_by_ids not implemented.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents)
        assert _sort_by_id(vectorstore.get_by_ids(ids)) == _sort_by_id(
            [
                Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
                Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            ]
        )

    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
        """Test that `add_documents` with existing IDs is idempotent.

        ??? note "Troubleshooting"

            If this test fails, check that `get_by_ids` is implemented and returns
            documents in the same order as the IDs passed in.

            This test also verifies that:

            1. IDs specified in the `Document.id` field are assigned when adding
                documents.
            2. If some documents include IDs and others don't string IDs are generated
                for the latter.

            !!! note
                `get_by_ids` was added to the `VectorStore` interface in
                `langchain-core` version 0.2.11. If difficult to implement, this
                test can be skipped by setting the `has_get_by_ids` property to
                `False`.

                ```python
                @property
                def has_get_by_ids(self) -> bool:
                    return False
                ```
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        if not self.has_get_by_ids:
            pytest.skip("get_by_ids not implemented.")

        documents = [
            Document(id="foo", page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents)
        assert "foo" in ids
        assert _sort_by_id(vectorstore.get_by_ids(ids)) == _sort_by_id(
            [
                Document(page_content="foo", metadata={"id": 1}, id="foo"),
                Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            ]
        )

    async def test_vectorstore_is_empty_async(self, vectorstore: VectorStore) -> None:
        """Test that the `VectorStore` is empty.

        ??? note "Troubleshooting"

            If this test fails, check that the test class (i.e., sub class of
            `VectorStoreIntegrationTests`) initializes an empty vector store in the
            `vectorestore` fixture.
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        assert await vectorstore.asimilarity_search("foo", k=1) == []

    async def test_add_documents_async(self, vectorstore: VectorStore) -> None:
        """Test adding documents into the `VectorStore`.

        ??? note "Troubleshooting"

            If this test fails, check that:

            1. We correctly initialize an empty vector store in the `vectorestore`
                fixture.
            2. Calling `.asimilarity_search` for the top `k` similar documents does
                not threshold by score.
            3. We do not mutate the original document object when adding it to the
                vector store (e.g., by adding an ID).
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        original_documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(original_documents)
        documents = await vectorstore.asimilarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
        ]

        # Verify that the original document object does not get mutated!
        # (e.g., an ID is added to the original document object)
        assert original_documents == [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]

    async def test_vectorstore_still_empty_async(
        self, vectorstore: VectorStore
    ) -> None:
        """Test that the `VectorStore` is still empty.

        This test should follow a test that adds documents.

        This just verifies that the fixture is set up properly to be empty
        after each test.

        ??? note "Troubleshooting"

            If this test fails, check that the test class (i.e., sub class of
            `VectorStoreIntegrationTests`) correctly clears the vector store in the
            `finally` block.
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        assert await vectorstore.asimilarity_search("foo", k=1) == []

    async def test_deleting_documents_async(self, vectorstore: VectorStore) -> None:
        """Test deleting documents from the `VectorStore`.

        ??? note "Troubleshooting"

            If this test fails, check that `aadd_documents` preserves identifiers
            passed in through `ids`, and that `delete` correctly removes
            documents.
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents, ids=["1", "2"])
        assert ids == ["1", "2"]
        await vectorstore.adelete(["1"])
        documents = await vectorstore.asimilarity_search("foo", k=1)
        assert documents == [Document(page_content="bar", metadata={"id": 2}, id="2")]

    async def test_deleting_bulk_documents_async(
        self, vectorstore: VectorStore
    ) -> None:
        """Test that we can delete several documents at once.

        ??? note "Troubleshooting"

            If this test fails, check that `adelete` correctly removes multiple
            documents when given a list of IDs.
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
            Document(page_content="baz", metadata={"id": 3}),
        ]

        await vectorstore.aadd_documents(documents, ids=["1", "2", "3"])
        await vectorstore.adelete(["1", "2"])
        documents = await vectorstore.asimilarity_search("foo", k=1)
        assert documents == [Document(page_content="baz", metadata={"id": 3}, id="3")]

    async def test_delete_missing_content_async(self, vectorstore: VectorStore) -> None:
        """Deleting missing content should not raise an exception.

        ??? note "Troubleshooting"

            If this test fails, check that `adelete` does not raise an exception
            when deleting IDs that do not exist.
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        await vectorstore.adelete(["1"])
        await vectorstore.adelete(["1", "2", "3"])

    async def test_add_documents_with_ids_is_idempotent_async(
        self, vectorstore: VectorStore
    ) -> None:
        """Adding by ID should be idempotent.

        ??? note "Troubleshooting"

            If this test fails, check that adding the same document twice with the
            same IDs has the same effect as adding it once (i.e., it does not
            duplicate the documents).
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        await vectorstore.aadd_documents(documents, ids=["1", "2"])
        await vectorstore.aadd_documents(documents, ids=["1", "2"])
        documents = await vectorstore.asimilarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id="2"),
            Document(page_content="foo", metadata={"id": 1}, id="1"),
        ]

    async def test_add_documents_by_id_with_mutation_async(
        self, vectorstore: VectorStore
    ) -> None:
        """Test that we can overwrite by ID using `add_documents`.

        ??? note "Troubleshooting"

            If this test fails, check that when `aadd_documents` is called with an
            ID that already exists in the vector store, the content is updated
            rather than duplicated.
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]

        await vectorstore.aadd_documents(documents=documents, ids=["1", "2"])

        # Now over-write content of ID 1
        new_documents = [
            Document(
                page_content="new foo", metadata={"id": 1, "some_other_field": "foo"}
            ),
        ]

        await vectorstore.aadd_documents(documents=new_documents, ids=["1"])

        # Check that the content has been updated
        documents = await vectorstore.asimilarity_search("new foo", k=2)
        assert documents == [
            Document(
                id="1",
                page_content="new foo",
                metadata={"id": 1, "some_other_field": "foo"},
            ),
            Document(id="2", page_content="bar", metadata={"id": 2}),
        ]

    async def test_get_by_ids_async(self, vectorstore: VectorStore) -> None:
        """Test get by IDs.

        This test requires that `get_by_ids` be implemented on the vector store.

        ??? note "Troubleshooting"

            If this test fails, check that `get_by_ids` is implemented and returns
            documents in the same order as the IDs passed in.

            !!! note
                `get_by_ids` was added to the `VectorStore` interface in
                `langchain-core` version 0.2.11. If difficult to implement, this
                test can be skipped by setting the `has_get_by_ids` property to
                `False`.

                ```python
                @property
                def has_get_by_ids(self) -> bool:
                    return False
                ```
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        if not self.has_get_by_ids:
            pytest.skip("get_by_ids not implemented.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents, ids=["1", "2"])
        retrieved_documents = await vectorstore.aget_by_ids(ids)
        assert _sort_by_id(retrieved_documents) == _sort_by_id(
            [
                Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
                Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            ]
        )

    async def test_get_by_ids_missing_async(self, vectorstore: VectorStore) -> None:
        """Test get by IDs with missing IDs.

        ??? note "Troubleshooting"

            If this test fails, check that `get_by_ids` is implemented and does not
            raise an exception when given IDs that do not exist.

            !!! note
                `get_by_ids` was added to the `VectorStore` interface in
                `langchain-core` version 0.2.11. If difficult to implement, this
                test can be skipped by setting the `has_get_by_ids` property to
                `False`.

                ```python
                @property
                def has_get_by_ids(self) -> bool:
                    return False
                ```
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        if not self.has_get_by_ids:
            pytest.skip("get_by_ids not implemented.")

        # This should not raise an exception
        assert await vectorstore.aget_by_ids(["1", "2", "3"]) == []

    async def test_add_documents_documents_async(
        self, vectorstore: VectorStore
    ) -> None:
        """Run `add_documents` tests.

        ??? note "Troubleshooting"

            If this test fails, check that `get_by_ids` is implemented and returns
            documents in the same order as the IDs passed in.

            Check also that `aadd_documents` will correctly generate string IDs if
            none are provided.

            !!! note
                `get_by_ids` was added to the `VectorStore` interface in
                `langchain-core` version 0.2.11. If difficult to implement, this
                test can be skipped by setting the `has_get_by_ids` property to
                `False`.

                ```python
                @property
                def has_get_by_ids(self) -> bool:
                    return False
                ```
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        if not self.has_get_by_ids:
            pytest.skip("get_by_ids not implemented.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents)
        assert _sort_by_id(await vectorstore.aget_by_ids(ids)) == _sort_by_id(
            [
                Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
                Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            ]
        )

    async def test_add_documents_with_existing_ids_async(
        self, vectorstore: VectorStore
    ) -> None:
        """Test that `add_documents` with existing IDs is idempotent.

        ??? note "Troubleshooting"

            If this test fails, check that `get_by_ids` is implemented and returns
            documents in the same order as the IDs passed in.

            This test also verifies that:

            1. IDs specified in the `Document.id` field are assigned when adding
                documents.
            2. If some documents include IDs and others don't string IDs are generated
                for the latter.

            !!! note
                `get_by_ids` was added to the `VectorStore` interface in
                `langchain-core` version 0.2.11. If difficult to implement, this
                test can be skipped by setting the `has_get_by_ids` property to
                `False`.

                ```python
                @property
                def has_get_by_ids(self) -> bool:
                    return False
                ```
        """
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        if not self.has_get_by_ids:
            pytest.skip("get_by_ids not implemented.")

        documents = [
            Document(id="foo", page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents)
        assert "foo" in ids
        assert _sort_by_id(await vectorstore.aget_by_ids(ids)) == _sort_by_id(
            [
                Document(page_content="foo", metadata={"id": 1}, id="foo"),
                Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            ]
        )
