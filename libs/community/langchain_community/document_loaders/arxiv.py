from typing import Any, Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.arxiv import ArxivAPIWrapper


class ArxivLoader(BaseLoader):
    """Load a query result from `Arxiv`.
    The loader converts the original PDF format into the text.

    Setup:
        Install ``arxiv`` and ``PyMuPDF`` packages.
        ``PyMuPDF`` transforms PDF files downloaded from the arxiv.org site
        into the text format.

        .. code-block:: bash

            pip install -U arxiv pymupdf


    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import ArxivLoader

            loader = ArxivLoader(
                query="reasoning",
                # load_max_docs=2,
                # load_all_available_meta=False
            )

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python
            Understanding the Reasoning Ability of Language Models
            From the Perspective of Reasoning Paths Aggre
            {
                'Published': '2024-02-29',
                'Title': 'Understanding the Reasoning Ability of Language Models From the
                        Perspective of Reasoning Paths Aggregation',
                'Authors': 'Xinyi Wang, Alfonso Amayuelas, Kexun Zhang, Liangming Pan,
                        Wenhu Chen, William Yang Wang',
                'Summary': 'Pre-trained language models (LMs) are able to perform complex reasoning
                        without explicit fine-tuning...'
            }


    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Understanding the Reasoning Ability of Language Models
            From the Perspective of Reasoning Paths Aggre
            {
                'Published': '2024-02-29',
                'Title': 'Understanding the Reasoning Ability of Language Models From the
                        Perspective of Reasoning Paths Aggregation',
                'Authors': 'Xinyi Wang, Alfonso Amayuelas, Kexun Zhang, Liangming Pan,
                        Wenhu Chen, William Yang Wang',
                'Summary': 'Pre-trained language models (LMs) are able to perform complex reasoning
                        without explicit fine-tuning...'
            }

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Understanding the Reasoning Ability of Language Models
            From the Perspective of Reasoning Paths Aggre
            {
                'Published': '2024-02-29',
                'Title': 'Understanding the Reasoning Ability of Language Models From the
                        Perspective of Reasoning Paths Aggregation',
                'Authors': 'Xinyi Wang, Alfonso Amayuelas, Kexun Zhang, Liangming Pan,
                        Wenhu Chen, William Yang Wang',
                'Summary': 'Pre-trained language models (LMs) are able to perform complex reasoning
                        without explicit fine-tuning...'
            }

    Use summaries of articles as docs:
        .. code-block:: python

            from langchain_community.document_loaders import ArxivLoader

            loader = ArxivLoader(
                query="reasoning"
            )

            docs = loader.get_summaries_as_docs()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Pre-trained language models (LMs) are able to perform complex reasoning
            without explicit fine-tuning
            {
                'Entry ID': 'http://arxiv.org/abs/2402.03268v2',
                'Published': datetime.date(2024, 2, 29),
                'Title': 'Understanding the Reasoning Ability of Language Models From the
                        Perspective of Reasoning Paths Aggregation',
                'Authors': 'Xinyi Wang, Alfonso Amayuelas, Kexun Zhang, Liangming Pan,
                        Wenhu Chen, William Yang Wang'
            }
    """  # noqa: E501

    def __init__(
        self, query: str, doc_content_chars_max: Optional[int] = None, **kwargs: Any
    ):
        """Initialize with search query to find documents in the Arxiv.
        Supports all arguments of `ArxivAPIWrapper`.

        Args:
            query: free text which used to find documents in the Arxiv
            doc_content_chars_max: cut limit for the length of a document's content
        """  # noqa: E501

        self.query = query
        self.client = ArxivAPIWrapper(
            doc_content_chars_max=doc_content_chars_max, **kwargs
        )

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load Arvix documents"""
        yield from self.client.lazy_load(self.query)

    def get_summaries_as_docs(self) -> List[Document]:
        """Uses papers summaries as documents rather than source Arvix papers"""
        return self.client.get_summaries_as_docs(self.query)
