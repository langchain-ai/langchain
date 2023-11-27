import hashlib
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
from unittest.mock import call

import pytest
from pytest_mock import MockerFixture

from langchain.document_transformers.document_transformers import (
    DocumentTransformers,
)

# from langchain_core.documents import Document
# from langchain_core.embeddings import Embeddings
# from langchain_core.vectorstores import VectorStore
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.rag_vectorstore import RAGVectorStore
from tests.unit_tests.document_transformers.sample_transformer import (
    LowerLazyTransformer,
)
from tests.unit_tests.document_transformers.test_runnable_transformers import (
    UpperLazyTransformer,
)

VST = TypeVar("VST", bound="VectorStore")


class FakeUUID:
    def __init__(self, prefix: str):
        self.uuid_count = 0
        self.prefix = prefix

    def __call__(self) -> str:
        self.uuid_count += 1
        return f"{self.prefix}{self.uuid_count:0>2}"


def _must_be_called(
    must_be_called: List[Tuple[List[str], List[Dict[str, Any]]]]
) -> List[Any]:
    calls = []
    for page_contents, metadatas in must_be_called:
        for page_content, metadata in zip(page_contents, metadatas):
            calls.append(
                call(
                    [
                        Document(page_content=page_content.upper(), metadata=metadata),
                        Document(page_content=page_content.lower(), metadata=metadata),
                    ]
                )
            )
    return calls


class FakeVectorStore(VectorStore):
    """Simulation of a vectorstore (without embedding)"""

    # def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
    #     print(f"add_documents({documents=},{kwargs=})")
    #     return super().add_documents(documents=documents,**kwargs)

    def __init__(self) -> None:
        self.uuid = FakeUUID(prefix="Fake-VS-")
        self.docs: Dict[str, List[Document]] = {}

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        print(f"add_documents({documents=},{kwargs=})")
        uuids = []
        for doc in documents:
            uuid = self.uuid()
            for word in doc.page_content.split(" "):
                word = word.lower()
                list_of_doc = self.docs.get(word, [])
                list_of_doc.append(doc)
                self.docs[word] = list_of_doc
            uuids.append(uuid)
        return uuids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return [str(self.uuid()) for _ in texts]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        result = {}  # Identity set
        for word in query.split(" "):
            word = word.lower()
            if word in self.docs:
                for doc in self.docs[word]:
                    result[id(doc)] = doc
        return list(result.values())

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        docs = self.similarity_search(query=query, k=k)
        len_docs = len(docs)
        c = 1 / len_docs
        return [(doc, (len_docs - i) * c) for i, doc in enumerate(docs)]

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        docs = self.similarity_search(query=query, k=k)
        len_docs = len(docs)
        c = 1 / len_docs
        return [(doc, (len_docs - i) * c) for i, doc in enumerate(docs)]

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        store = cls()
        return store

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        return True


class SplitterWithUniqId(CharacterTextSplitter):
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        documents = super().split_documents(documents)
        for doc in documents:
            doc.metadata[
                "split_id"
            ] = f'{doc.metadata["id"]}-{doc.metadata["start_index"]}'
        return documents


parent_transformer = SplitterWithUniqId(
    chunk_size=1,
    chunk_overlap=0,
    separator=" ",
    add_start_index=True,
)

chunk_transformer = DocumentTransformers(
    transformers=[
        UpperLazyTransformer(),
        LowerLazyTransformer(),
    ]
)


# %% parent_and_chunk_transformer


def test_parent_and_chunk_transformer(mocker: MockerFixture) -> None:
    """
    parent_transformer = True
    chunk_transformer = True
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})

    # ----
    ids = vs.add_documents([doc1, doc2])
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 4
    assert ids == [
        hashlib.sha256(
            str(doc1.metadata[vs.source_id_key]).encode("utf-8")
        ).hexdigest(),
        hashlib.sha256(
            str(doc2.metadata[vs.source_id_key]).encode("utf-8")
        ).hexdigest(),
    ]
    spy_add_documents.assert_has_calls(
        _must_be_called(
            [
                (
                    ["HELLO"],
                    [
                        {
                            "id": 1,
                            "start_index": 0,
                            "split_id": "1-0",
                            vs.chunk_id_key: "chunk-01",
                        },
                    ],
                ),
                (
                    ["WORD"],
                    [
                        {
                            "id": 1,
                            "start_index": 6,
                            "split_id": "1-6",
                            vs.chunk_id_key: "chunk-02",
                        }
                    ],
                ),
                (
                    ["HAPPY"],
                    [
                        {
                            "id": 2,
                            "start_index": 0,
                            "split_id": "2-0",
                            vs.chunk_id_key: "chunk-03",
                        }
                    ],
                ),
                (
                    ["DAYS"],
                    [
                        {
                            "id": 2,
                            "start_index": 6,
                            "split_id": "2-6",
                            vs.chunk_id_key: "chunk-04",
                        }
                    ],
                ),
            ]
        )
    )
    spy_delete.assert_called_with(ids=list({f"Fake-VS-0{i}" for i in range(1, 9)}))


def test_parent_and_chunk_tranformer_childid(mocker: MockerFixture) -> None:
    """
    Sometime, the result of a parent transformation is a list of documents
    with an uniq id. It's not necessary to inject a new one.
    You can set the name of this id with `chunk_id_key`.

    parent_transformer = True
    chunk_transformer = True
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        chunk_id_key="split_id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})

    # ----
    ids = vs.add_documents([doc1, doc2])
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 4
    assert ids == [
        hashlib.sha256(
            str(doc1.metadata[vs.source_id_key]).encode("utf-8")
        ).hexdigest(),
        hashlib.sha256(
            str(doc2.metadata[vs.source_id_key]).encode("utf-8")
        ).hexdigest(),
    ]
    spy_add_documents.assert_has_calls(
        _must_be_called(
            [
                (
                    ["HELLO"],
                    [
                        {
                            "id": 1,
                            "start_index": 0,
                            "split_id": "1-0",
                        }
                    ],
                ),
                (
                    ["WORD"],
                    [
                        {
                            "id": 1,
                            "start_index": 6,
                            "split_id": "1-6",
                        }
                    ],
                ),
                (
                    ["HAPPY"],
                    [
                        {
                            "id": 2,
                            "start_index": 0,
                            "split_id": "2-0",
                        }
                    ],
                ),
                (
                    ["DAYS"],
                    [
                        {
                            "id": 2,
                            "start_index": 6,
                            "split_id": "2-6",
                        }
                    ],
                ),
            ]
        )
    )
    spy_delete.assert_called_with(list({f"Fake-VS-0{i}" for i in range(1, 9)}))


def test_parent_and_chunk_transformer_ids(mocker: MockerFixture) -> None:
    """
    parent_transformer = True
    chunk_transformer = True
    ids = Yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    force_ids = [fake_uuid(), fake_uuid()]

    # ----
    ids = vs.add_documents([doc1, doc2], ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 4
    assert ids == force_ids
    spy_add_documents.assert_has_calls(
        _must_be_called(
            [
                (
                    ["HELLO"],
                    [
                        {
                            "id": 1,
                            "start_index": 0,
                            "split_id": "1-0",
                            vs.chunk_id_key: "chunk-01",
                        },
                    ],
                ),
                (
                    ["WORD"],
                    [
                        {
                            "id": 1,
                            "start_index": 6,
                            "split_id": "1-6",
                            vs.chunk_id_key: "chunk-02",
                        }
                    ],
                ),
                (
                    ["HAPPY"],
                    [
                        {
                            "id": 2,
                            "start_index": 0,
                            "split_id": "2-0",
                            vs.chunk_id_key: "chunk-03",
                        }
                    ],
                ),
                (
                    ["DAYS"],
                    [
                        {
                            "id": 2,
                            "start_index": 6,
                            "split_id": "2-6",
                            vs.chunk_id_key: "chunk-04",
                        }
                    ],
                ),
            ]
        )
    )
    spy_delete.assert_called_with(ids=list({f"Fake-VS-0{i}" for i in range(1, 9)}))


# %% chunk_transformer
def test_chunk_transformer(mocker: MockerFixture) -> None:
    """
    parent_transformer = False
    chunk_transformer = True
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    split_docs = parent_transformer.split_documents([doc1, doc2])
    # ----

    ids = vs.add_documents(documents=split_docs)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 4
    spy_add_documents.assert_has_calls(
        _must_be_called(
            [
                (
                    ["HELLO"],
                    [
                        {
                            "id": 1,
                            "start_index": 0,
                            "split_id": "1-0",
                            vs.chunk_id_key: "chunk-01",
                        }
                    ],
                ),
                (
                    ["WORD"],
                    [
                        {
                            "id": 1,
                            "start_index": 6,
                            "split_id": "1-6",
                            vs.chunk_id_key: "chunk-02",
                        }
                    ],
                ),
                (
                    ["HAPPY"],
                    [
                        {
                            "id": 2,
                            "start_index": 0,
                            "split_id": "2-0",
                            vs.chunk_id_key: "chunk-03",
                        }
                    ],
                ),
                (
                    ["DAYS"],
                    [
                        {
                            "id": 2,
                            "start_index": 6,
                            "split_id": "2-6",
                            vs.chunk_id_key: "chunk-04",
                        }
                    ],
                ),
            ]
        )
    )
    spy_delete.assert_called_with(list({f"Fake-VS-0{i}" for i in range(1, 9)}))


def test_chunk_transformer_ids(mocker: MockerFixture) -> None:
    """
    parent_transformer = False
    chunk_transformer = True
    ids = Yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    split_docs = parent_transformer.split_documents([doc1, doc2])
    force_ids = [fake_uuid() for _ in range(0, len(split_docs))]
    # ----
    ids = vs.add_documents(documents=split_docs, ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 4
    assert ids == force_ids
    spy_add_documents.assert_has_calls(
        _must_be_called(
            [
                (
                    ["HELLO"],
                    [
                        {
                            "id": 1,
                            "start_index": 0,
                            "split_id": "1-0",
                            vs.chunk_id_key: "persistance-01",
                        }
                    ],
                ),
                (
                    ["WORD"],
                    [
                        {
                            "id": 1,
                            "start_index": 6,
                            "split_id": "1-6",
                            vs.chunk_id_key: "persistance-02",
                        }
                    ],
                ),
                (
                    ["HAPPY"],
                    [
                        {
                            "id": 2,
                            "start_index": 0,
                            "split_id": "2-0",
                            vs.chunk_id_key: "persistance-03",
                        }
                    ],
                ),
                (
                    ["DAYS"],
                    [
                        {
                            "id": 2,
                            "start_index": 6,
                            "split_id": "2-6",
                            vs.chunk_id_key: "persistance-04",
                        }
                    ],
                ),
            ]
        )
    )
    spy_delete.assert_called_with(list({f"Fake-VS-0{i}" for i in range(1, 9)}))


# %% parent_transformer
def test_parent_transformer(mocker: MockerFixture) -> None:
    """
    parent_transformer = True
    chunk_transformer = False
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    # ----
    ids = vs.add_documents(documents=[doc1, doc2])
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 1
    assert ids == [
        hashlib.sha256(
            str(doc1.metadata[vs.source_id_key]).encode("utf-8")
        ).hexdigest(),
        hashlib.sha256(
            str(doc2.metadata[vs.source_id_key]).encode("utf-8")
        ).hexdigest(),
    ]
    spy_add_documents.assert_has_calls(
        [
            call(
                documents=[
                    Document(
                        page_content="Hello",
                        metadata={"id": 1, "start_index": 0, "split_id": "1-0"},
                    ),
                    Document(
                        page_content="word",
                        metadata={"id": 1, "start_index": 6, "split_id": "1-6"},
                    ),
                    Document(
                        page_content="Happy",
                        metadata={"id": 2, "start_index": 0, "split_id": "2-0"},
                    ),
                    Document(
                        page_content="days",
                        metadata={"id": 2, "start_index": 6, "split_id": "2-6"},
                    ),
                ],
                ids=[f"chunk-0{i}" for i in range(1, 5)],
            )
        ]
    )
    spy_delete.assert_called_with(ids=[f"chunk-0{i}" for i in range(1, 5)])


def test_parent_transformer_ids(mocker: MockerFixture) -> None:
    """
    parent_transformer = True
    chunk_transformer = False
    ids = Yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    docs = [doc1, doc2]
    force_ids = [fake_uuid() for _ in range(0, len(docs))]
    # ----
    ids = vs.add_documents(documents=docs, ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 1
    assert ids == force_ids
    spy_add_documents.assert_has_calls(
        [
            call(
                documents=[
                    Document(
                        page_content="Hello",
                        metadata={"id": 1, "start_index": 0, "split_id": "1-0"},
                    ),
                    Document(
                        page_content="word",
                        metadata={"id": 1, "start_index": 6, "split_id": "1-6"},
                    ),
                    Document(
                        page_content="Happy",
                        metadata={"id": 2, "start_index": 0, "split_id": "2-0"},
                    ),
                    Document(
                        page_content="days",
                        metadata={"id": 2, "start_index": 6, "split_id": "2-6"},
                    ),
                ],
                ids=[f"chunk-0{i}" for i in range(1, 5)],
            )
        ]
    )
    spy_delete.assert_called_with(ids=[f"chunk-0{i}" for i in range(1, 5)])


# %% without_transformer
def test_without_transformer(mocker: MockerFixture) -> None:
    """
    parent_transformer = False
    chunk_transformer = False
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    split_docs = parent_transformer.split_documents([doc1, doc2])
    # ----

    # RunnableGenerator(
    #     transform=partial(_transform_documents_generator,
    #                       transformers=self.transformers)
    # ).transform([doc1])

    ids = vs.add_documents(documents=split_docs)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 1
    spy_add_documents.assert_has_calls(
        [
            call(
                documents=[
                    Document(
                        page_content="Hello",
                        metadata={"id": 1, "start_index": 0, "split_id": "1-0"},
                    ),
                    Document(
                        page_content="word",
                        metadata={"id": 1, "start_index": 6, "split_id": "1-6"},
                    ),
                    Document(
                        page_content="Happy",
                        metadata={"id": 2, "start_index": 0, "split_id": "2-0"},
                    ),
                    Document(
                        page_content="days",
                        metadata={"id": 2, "start_index": 6, "split_id": "2-6"},
                    ),
                ],
                ids=[f"chunk-0{i}" for i in range(1, 5)],
            )
        ]
    )
    spy_delete.assert_called_with(ids=[f"chunk-0{i}" for i in range(1, 5)])


def test_without_transformer_ids(mocker: MockerFixture) -> None:
    """
    parent_transformer = False
    chunk_transformer = False
    ids = yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, "add_documents")
    spy_delete = mocker.spy(fake_vs, "delete")
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    split_docs = parent_transformer.split_documents([doc1, doc2])
    force_ids = [fake_uuid() for _ in range(0, len(split_docs))]
    # ----
    ids = vs.add_documents(documents=split_docs, ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == "word"
    assert spy_add_documents.call_count == 1
    assert ids == force_ids
    spy_add_documents.assert_has_calls(
        [
            call(
                documents=[
                    Document(
                        page_content="Hello",
                        metadata={"id": 1, "start_index": 0, "split_id": "1-0"},
                    ),
                    Document(
                        page_content="word",
                        metadata={"id": 1, "start_index": 6, "split_id": "1-6"},
                    ),
                    Document(
                        page_content="Happy",
                        metadata={"id": 2, "start_index": 0, "split_id": "2-0"},
                    ),
                    Document(
                        page_content="days",
                        metadata={"id": 2, "start_index": 6, "split_id": "2-6"},
                    ),
                ],
                ids=[f"persistance-0{i}" for i in range(1, 5)],
            )
        ]
    )
    spy_delete.assert_called_with(ids=[f"persistance-0{i}" for i in range(1, 5)])


# %% search
def test_search_without_parent_transformer(mocker: MockerFixture) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_search = mocker.spy(fake_vs, "search")
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.search("hello", search_type="similarity", k=2)
    # ----
    assert len(result) == 2
    assert result[0].page_content == "Hello word"
    assert result[1].page_content == "Hello langchain"
    spy_search.assert_called_with(query="hello", search_type="similarity", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asearch_without_parent_transformer(mocker: MockerFixture) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_search = mocker.spy(fake_vs, "search")
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asearch("hello", search_type="similarity", k=2)
    # ----
    assert len(result) == 2
    assert result[0].page_content == "Hello word"
    assert result[1].page_content == "Hello langchain"
    spy_search.assert_called_with(query="hello", search_type="similarity", k=10)


# %% similarity_search
def test_similarity_search_without_parent_transformer(mocker: MockerFixture) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search = mocker.spy(fake_vs, "similarity_search")
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search("hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0].page_content == "Hello word"
    assert result[1].page_content == "Hello langchain"
    spy_similarity_search.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_without_transformer(mocker: MockerFixture) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search = mocker.spy(fake_vs, "similarity_search")
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search("hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0].page_content == "Hello word"
    assert result[1].page_content == "Hello langchain"
    spy_similarity_search.assert_called_with(query="hello", k=10)


# %% similarity_search_with_score
def test_similarity_search_with_score_without_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_score = mocker.spy(
        fake_vs, "similarity_search_with_score"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search_with_score(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello word"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello langchain"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_score.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_with_score_without_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_score = mocker.spy(
        fake_vs, "similarity_search_with_score"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search_with_score(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello word"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello langchain"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_score.assert_called_with(query="hello", k=10)


def test_similarity_search_with_score_with_parent_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_score = mocker.spy(
        fake_vs, "similarity_search_with_score"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search_with_score(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello"
    assert result[0][0].metadata["id"] == 1
    assert result[1][0].page_content == "Hello"
    assert result[1][0].metadata["id"] == 3
    spy_similarity_search_with_score.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_with_score_with_parent_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_score = mocker.spy(
        fake_vs, "similarity_search_with_score"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search_with_score(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello"
    assert result[0][0].metadata["id"] == 1
    assert result[1][0].page_content == "Hello"
    assert result[1][0].metadata["id"] == 3
    spy_similarity_search_with_score.assert_called_with(query="hello", k=10)


def test_similarity_search_with_score_with_chunk_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_score = mocker.spy(
        fake_vs, "similarity_search_with_score"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search_with_score(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello word"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello langchain"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_score.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_with_score_with_chunk_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_score = mocker.spy(
        fake_vs, "similarity_search_with_score"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search_with_score(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello word"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello langchain"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_score.assert_called_with(query="hello", k=10)


def test_similarity_search_with_score(mocker: MockerFixture) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_score = mocker.spy(
        fake_vs, "similarity_search_with_score"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search_with_score(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_score.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_with_score(mocker: MockerFixture) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_score = mocker.spy(
        fake_vs, "similarity_search_with_score"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search_with_score(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_score.assert_called_with(query="hello", k=10)


# %% similarity_search_with_relevance_scores
def test_similarity_search_with_relevance_scores_without_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_relevance_scores = mocker.spy(
        fake_vs, "similarity_search_with_relevance_scores"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search_with_relevance_scores(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello word"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello langchain"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_relevance_scores.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_with_relevance_scores_without_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_relevance_scores = mocker.spy(
        fake_vs, "similarity_search_with_relevance_scores"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search_with_relevance_scores(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello word"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello langchain"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_relevance_scores.assert_called_with(query="hello", k=10)


def test_similarity_search_with_relevance_scores_with_parent_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_relevance_scores = mocker.spy(
        fake_vs, "similarity_search_with_relevance_scores"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search_with_relevance_scores(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello"
    assert result[0][0].metadata["id"] == 1
    assert result[1][0].page_content == "Hello"
    assert result[1][0].metadata["id"] == 3
    spy_similarity_search_with_relevance_scores.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_with_relevance_scores_with_parent_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_relevance_scores = mocker.spy(
        fake_vs, "similarity_search_with_relevance_scores"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search_with_relevance_scores(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello"
    assert result[0][0].metadata["id"] == 1
    assert result[1][0].page_content == "Hello"
    assert result[1][0].metadata["id"] == 3
    spy_similarity_search_with_relevance_scores.assert_called_with(query="hello", k=10)


def test_similarity_search_with_relevance_scores_with_chunk_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_relevance_scores = mocker.spy(
        fake_vs, "similarity_search_with_relevance_scores"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search_with_relevance_scores(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello word"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello langchain"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_relevance_scores.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_with_relevance_scores_with_chunk_transformer(
    mocker: MockerFixture,
) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_relevance_scores = mocker.spy(
        fake_vs, "similarity_search_with_relevance_scores"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search_with_relevance_scores(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello word"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello langchain"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_relevance_scores.assert_called_with(query="hello", k=10)


def test_similarity_search_with_relevance_scores(mocker: MockerFixture) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_relevance_scores = mocker.spy(
        fake_vs, "similarity_search_with_relevance_scores"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    vs.add_documents([doc1, doc2, doc3, doc4])
    # ----
    result = vs.similarity_search_with_relevance_scores(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_relevance_scores.assert_called_with(query="hello", k=10)


@pytest.mark.asyncio
@pytest.mark.skip(reason="async not implemented yet")
async def test_asimilarity_search_with_relevance_scores(mocker: MockerFixture) -> None:
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_similarity_search_with_relevance_scores = mocker.spy(
        fake_vs, "similarity_search_with_relevance_scores"
    )
    vs = RAGVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        source_id_key="id",
        search_kwargs={"k": 10},
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    doc3 = Document(page_content="Hello langchain", metadata={"id": 3})
    doc4 = Document(page_content="Hello llm", metadata={"id": 4})
    await vs.aadd_documents([doc1, doc2, doc3, doc4])
    # ----
    result = await vs.asimilarity_search_with_relevance_scores(query="hello", k=2)
    # ----
    assert len(result) == 2
    assert result[0][0].page_content == "Hello"
    assert result[0][0].metadata["id"] == 1
    assert result[0][1] == 1.0
    assert result[1][0].page_content == "Hello"
    assert result[1][0].metadata["id"] == 3
    assert result[1][1] == (1.0 / 3) * 2
    spy_similarity_search_with_relevance_scores.assert_called_with(query="hello", k=10)
