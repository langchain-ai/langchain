from typing import Any, Dict, Iterable, List, Optional, Set, Union

from langchain_core._api import beta
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor,
)
from langchain_community.graph_vectorstores.links import Link

# TypeAlias is not available in Python 3.9, we can't use that or the newer `type`.
GLiNERInput = Union[str, Document]


@beta()
class GLiNERLinkExtractor(LinkExtractor[GLiNERInput]):
    """Link documents with common named entities using `GLiNER`_.

    `GLiNER`_ is a Named Entity Recognition (NER) model capable of identifying any
    entity type using a bidirectional transformer encoder (BERT-like).

    The ``GLiNERLinkExtractor`` uses GLiNER to create links between documents that
    have named entities in common.

    Example::

        extractor = GLiNERLinkExtractor(
            labels=["Person", "Award", "Date", "Competitions", "Teams"]
        )
        results = extractor.extract_one("some long text...")

    .. _GLiNER: https://github.com/urchade/GLiNER

    .. seealso::

            - :mod:`How to use a graph vector store <langchain_community.graph_vectorstores>`
            - :class:`How to create links between documents <langchain_community.graph_vectorstores.links.Link>`

    How to link Documents on common named entities
    ==============================================

    Preliminaries
    -------------

    Install the ``gliner`` package:

    .. code-block:: bash

        pip install -q langchain_community gliner

    Usage
    -----

    We load the ``state_of_the_union.txt`` file, chunk it, then for each chunk we
    extract named entity links and add them to the chunk.

    Using extract_one()
    ^^^^^^^^^^^^^^^^^^^

    We can use :meth:`extract_one` on a document to get the links and add the links
    to the document metadata with
    :meth:`~langchain_community.graph_vectorstores.links.add_links`::

        from langchain_community.document_loaders import TextLoader
        from langchain_community.graph_vectorstores import CassandraGraphVectorStore
        from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
        from langchain_community.graph_vectorstores.links import add_links
        from langchain_text_splitters import CharacterTextSplitter

        loader = TextLoader("state_of_the_union.txt")
        raw_documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        ner_extractor = GLiNERLinkExtractor(["Person", "Topic"])
        for document in documents:
            links = ner_extractor.extract_one(document)
            add_links(document, links)

        print(documents[0].metadata)

    .. code-block:: output

        {'source': 'state_of_the_union.txt', 'links': [Link(kind='entity:Person', direction='bidir', tag='President Zelenskyy'), Link(kind='entity:Person', direction='bidir', tag='Vladimir Putin')]}

    Using LinkExtractorTransformer
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Using the :class:`~langchain_community.graph_vectorstores.extractors.link_extractor_transformer.LinkExtractorTransformer`,
    we can simplify the link extraction::

        from langchain_community.document_loaders import TextLoader
        from langchain_community.graph_vectorstores.extractors import (
            GLiNERLinkExtractor,
            LinkExtractorTransformer,
        )
        from langchain_text_splitters import CharacterTextSplitter

        loader = TextLoader("state_of_the_union.txt")
        raw_documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        ner_extractor = GLiNERLinkExtractor(["Person", "Topic"])
        transformer = LinkExtractorTransformer([ner_extractor])
        documents = transformer.transform_documents(documents)

        print(documents[0].metadata)

    .. code-block:: output

        {'source': 'state_of_the_union.txt', 'links': [Link(kind='entity:Person', direction='bidir', tag='President Zelenskyy'), Link(kind='entity:Person', direction='bidir', tag='Vladimir Putin')]}

    The documents with named entity links can then be added to a :class:`~langchain_community.graph_vectorstores.base.GraphVectorStore`::

        from langchain_community.graph_vectorstores import CassandraGraphVectorStore

        store = CassandraGraphVectorStore.from_documents(documents=documents, embedding=...)

    Args:
        labels: List of kinds of entities to extract.
        kind: Kind of links to produce with this extractor.
        model: GLiNER model to use.
        extract_kwargs: Keyword arguments to pass to GLiNER.
    """  # noqa: E501

    def __init__(
        self,
        labels: List[str],
        *,
        kind: str = "entity",
        model: str = "urchade/gliner_mediumv2.1",
        extract_kwargs: Optional[Dict[str, Any]] = None,
    ):
        try:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained(model)

        except ImportError:
            raise ImportError(
                "gliner is required for GLiNERLinkExtractor. "
                "Please install it with `pip install gliner`."
            ) from None

        self._labels = labels
        self._kind = kind
        self._extract_kwargs = extract_kwargs or {}

    def extract_one(self, input: GLiNERInput) -> Set[Link]:  # noqa: A002
        return next(iter(self.extract_many([input])))

    def extract_many(
        self,
        inputs: Iterable[GLiNERInput],
    ) -> Iterable[Set[Link]]:
        strs = [i if isinstance(i, str) else i.page_content for i in inputs]
        for entities in self._model.batch_predict_entities(
            strs, self._labels, **self._extract_kwargs
        ):
            yield {
                Link.bidir(kind=f"{self._kind}:{e['label']}", tag=e["text"])
                for e in entities
            }
