from typing import Any, Dict, Iterable, Optional, Set, Union

from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.graph_vectorstores.links import Link

from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor,
)

KeybertInput = Union[str, Document]


@beta()
class KeybertLinkExtractor(LinkExtractor[KeybertInput]):
    def __init__(
        self,
        *,
        kind: str = "kw",
        embedding_model: str = "all-MiniLM-L6-v2",
        extract_keywords_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Extract keywords using `KeyBERT <https://maartengr.github.io/KeyBERT/>`_.

        KeyBERT is a minimal and easy-to-use keyword extraction technique that
        leverages BERT embeddings to create keywords and keyphrases that are most
        similar to a document.

        The KeybertLinkExtractor uses KeyBERT to create links between documents that
        have keywords in common.

        Example::

            extractor = KeybertLinkExtractor()
            results = extractor.extract_one("lorem ipsum...")

        .. seealso::

            - :mod:`How to use a graph vector store <langchain_community.graph_vectorstores>`
            - :class:`How to create links between documents <langchain_core.graph_vectorstores.links.Link>`

        How to link Documents on common keywords using Keybert
        ======================================================

        Preliminaries
        -------------

        Install the keybert package:

        .. code-block:: bash

            pip install -q langchain_community keybert

        Usage
        -----

        We load the ``state_of_the_union.txt`` file, chunk it, then for each chunk we
        extract keyword links and add them to the chunk.

        Using extract_one()
        ^^^^^^^^^^^^^^^^^^^

        We can use :meth:`extract_one` on a document to get the links and add the links
        to the document metadata with
        :meth:`~langchain_core.graph_vectorstores.links.add_links`::

            from langchain_community.document_loaders import TextLoader
            from langchain_community.graph_vectorstores import CassandraGraphVectorStore
            from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
            from langchain_core.graph_vectorstores.links import add_links
            from langchain_text_splitters import CharacterTextSplitter

            loader = TextLoader("state_of_the_union.txt")

            raw_documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

            documents = text_splitter.split_documents(raw_documents)
            keyword_extractor = KeybertLinkExtractor()

            for document in documents:
                links = keyword_extractor.extract_one(document)
                add_links(document, links)

            print(documents[0].metadata)

        .. code-block:: output

            {'source': 'state_of_the_union.txt', 'links': [Link(kind='kw', direction='bidir', tag='ukraine'), Link(kind='kw', direction='bidir', tag='ukrainian'), Link(kind='kw', direction='bidir', tag='putin'), Link(kind='kw', direction='bidir', tag='vladimir'), Link(kind='kw', direction='bidir', tag='russia')]}

        Using LinkExtractorTransformer
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        Using the :class:`~langchain_community.graph_vectorstores.extractors.keybert_link_extractor.LinkExtractorTransformer`,
        we can simplify the link extraction::

            from langchain_community.document_loaders import TextLoader
            from langchain_community.graph_vectorstores.extractors import (
                KeybertLinkExtractor,
                LinkExtractorTransformer,
            )
            from langchain_text_splitters import CharacterTextSplitter

            loader = TextLoader("state_of_the_union.txt")
            raw_documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(raw_documents)

            transformer = LinkExtractorTransformer([KeybertLinkExtractor()])
            documents = transformer.transform_documents(documents)

            print(documents[0].metadata)

        .. code-block:: output

            {'source': 'state_of_the_union.txt', 'links': [Link(kind='kw', direction='bidir', tag='ukraine'), Link(kind='kw', direction='bidir', tag='ukrainian'), Link(kind='kw', direction='bidir', tag='putin'), Link(kind='kw', direction='bidir', tag='vladimir'), Link(kind='kw', direction='bidir', tag='russia')]}

        The documents with keyword links can then be added to a :class:`~langchain_core.graph_vectorstores.base.GraphVectorStore`::

            from langchain_community.graph_vectorstores import CassandraGraphVectorStore

            store = CassandraGraphVectorStore.from_documents(documents=documents, embedding=...)

        Args:
            kind: Kind of links to produce with this extractor.
            embedding_model: Name of the embedding model to use with KeyBERT.
            extract_keywords_kwargs: Keyword arguments to pass to KeyBERT's
                ``extract_keywords`` method.
        """  # noqa: E501
        try:
            import keybert

            self._kw_model = keybert.KeyBERT(model=embedding_model)
        except ImportError:
            raise ImportError(
                "keybert is required for KeybertLinkExtractor. "
                "Please install it with `pip install keybert`."
            ) from None

        self._kind = kind
        self._extract_keywords_kwargs = extract_keywords_kwargs or {}

    def extract_one(self, input: KeybertInput) -> Set[Link]:  # noqa: A002
        keywords = self._kw_model.extract_keywords(
            input if isinstance(input, str) else input.page_content,
            **self._extract_keywords_kwargs,
        )
        return {Link.bidir(kind=self._kind, tag=kw[0]) for kw in keywords}

    def extract_many(
        self,
        inputs: Iterable[KeybertInput],
    ) -> Iterable[Set[Link]]:
        inputs = list(inputs)
        if len(inputs) == 1:
            # Even though we pass a list, if it contains one item, keybert will
            # flatten it. This means it's easier to just call the special case
            # for one item.
            yield self.extract_one(inputs[0])
        elif len(inputs) > 1:
            strs = [i if isinstance(i, str) else i.page_content for i in inputs]
            extracted = self._kw_model.extract_keywords(
                strs, **self._extract_keywords_kwargs
            )
            for keywords in extracted:
                yield {Link.bidir(kind=self._kind, tag=kw[0]) for kw in keywords}
