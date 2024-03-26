from typing import Any, Iterable, List, Optional, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.vectorstores import VectorStore


class DummyVectorStore(VectorStore):
    def __init__(self, init_arg: Optional[str] = None):
        self.texts: List[str] = []
        self.metadatas: List[dict] = []
        self._embeddings: Optional[Embeddings] = None
        self.init_arg = init_arg

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        self.texts.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)
        return ["dummy_id"]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [
            Document(page_content=query, metadata={"metadata": query, "other": "other"})
        ] * k

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "DummyVectorStore":
        store = DummyVectorStore(**kwargs)
        store.add_texts(texts, metadatas)
        store._embeddings = embedding
        return store


def test_add_example() -> None:
    vector_store = DummyVectorStore()
    selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_store, input_keys=["foo", "foo3"]
    )
    selector.add_example({"foo": "bar", "foo2": "bar2", "foo3": "bar3"})
    assert vector_store.texts == ["bar bar3"]
    assert vector_store.metadatas == [{"foo": "bar", "foo2": "bar2", "foo3": "bar3"}]


async def test_aadd_example() -> None:
    vector_store = DummyVectorStore()
    selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_store, input_keys=["foo", "foo3"]
    )
    await selector.aadd_example({"foo": "bar", "foo2": "bar2", "foo3": "bar3"})
    assert vector_store.texts == ["bar bar3"]
    assert vector_store.metadatas == [{"foo": "bar", "foo2": "bar2", "foo3": "bar3"}]


def test_select_examples() -> None:
    vector_store = DummyVectorStore()
    selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_store, input_keys=["foo2"], example_keys=["metadata"], k=2
    )
    examples = selector.select_examples({"foo": "bar", "foo2": "bar2"})
    assert examples == [{"metadata": "bar2"}] * 2


async def test_aselect_examples() -> None:
    vector_store = DummyVectorStore()
    selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_store, input_keys=["foo2"], example_keys=["metadata"], k=2
    )
    examples = await selector.aselect_examples({"foo": "bar", "foo2": "bar2"})
    assert examples == [{"metadata": "bar2"}] * 2


def test_from_examples() -> None:
    examples = [{"foo": "bar"}]
    embeddings = FakeEmbeddings(size=1)
    selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=DummyVectorStore,
        k=2,
        input_keys=["foo"],
        example_keys=["some_example_key"],
        vectorstore_kwargs={"vs_foo": "vs_bar"},
        init_arg="some_init_arg",
    )
    assert selector.input_keys == ["foo"]
    assert selector.example_keys == ["some_example_key"]
    assert selector.k == 2
    assert selector.vectorstore_kwargs == {"vs_foo": "vs_bar"}

    assert isinstance(selector.vectorstore, DummyVectorStore)
    vector_store = cast(DummyVectorStore, selector.vectorstore)
    assert vector_store.embeddings is embeddings
    assert vector_store.init_arg == "some_init_arg"
    assert vector_store.texts == ["bar"]
    assert vector_store.metadatas == [{"foo": "bar"}]


async def test_afrom_examples() -> None:
    examples = [{"foo": "bar"}]
    embeddings = FakeEmbeddings(size=1)
    selector = await SemanticSimilarityExampleSelector.afrom_examples(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=DummyVectorStore,
        k=2,
        input_keys=["foo"],
        example_keys=["some_example_key"],
        vectorstore_kwargs={"vs_foo": "vs_bar"},
        init_arg="some_init_arg",
    )
    assert selector.input_keys == ["foo"]
    assert selector.example_keys == ["some_example_key"]
    assert selector.k == 2
    assert selector.vectorstore_kwargs == {"vs_foo": "vs_bar"}

    assert isinstance(selector.vectorstore, DummyVectorStore)
    vector_store = cast(DummyVectorStore, selector.vectorstore)
    assert vector_store.embeddings is embeddings
    assert vector_store.init_arg == "some_init_arg"
    assert vector_store.texts == ["bar"]
    assert vector_store.metadatas == [{"foo": "bar"}]
