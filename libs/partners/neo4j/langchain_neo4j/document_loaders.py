"""Neo4j document loader."""

from typing import Iterator

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class Neo4jLoader(BaseLoader):
    # TODO: Replace all TODOs in docstring. See example docstring:
    # https://github.com/langchain-ai/langchain/blob/869523ad728e6b76d77f170cce13925b4ebc3c1e/libs/community/langchain_community/document_loaders/recursive_url_loader.py#L54
    """
    Neo4j document loader integration

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``langchain-neo4j`` and set environment variable ``NEO4J_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-neo4j
            export NEO4J_API_KEY="your-api-key"

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import Neo4jLoader

            loader = Neo4jLoader(
                # required params = ...
                # other params = ...
            )

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

            TODO: Example output

    # TODO: Delete if async load is not implemented
    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            TODO: Example output
    """  # noqa: E501

    # TODO: This method must be implemented to load documents.
    # Do not implement load(), a default implementation is already available.
    def lazy_load(self) -> Iterator[Document]:
        raise NotImplementedError()

    # TODO: Implement if you would like to change default BaseLoader implementation
    # async def alazy_load(self) -> AsyncIterator[Document]:
