""".. title:: Graph Vector Store

Graph Vector Store
==================

Sometimes embedding models don't capture all the important relationships between
documents.
Graph Vector Stores are an extension to both vector stores and retrievers that allow
documents to be explicitly connected to each other.

Graph vector store retrievers use both vector similarity and links to find documents
related to an unstructured query.

Graphs allow linking between documents.
Each document identifies tags that link to and from it.
For example, a paragraph of text may be linked to URLs based on the anchor tags in
it's content and linked from the URL(s) it is published at.

`Link extractors <langchain_community.graph_vectorstores.extractors.link_extractor.LinkExtractor>`
can be used to extract links from documents.

Example::

    graph_vector_store = CassandraGraphVectorStore()
    link_extractor = HtmlLinkExtractor()
    links = link_extractor.extract_one(HtmlInput(document.page_content, "http://mysite"))
    add_links(document, links)
    graph_vector_store.add_document(document)

.. seealso::

    - :class:`How to use a graph vector store as a retriever <langchain_community.graph_vectorstores.base.GraphVectorStoreRetriever>`
    - :class:`How to create links between documents <langchain_community.graph_vectorstores.links.Link>`
    - :class:`How to link Documents on hyperlinks in HTML <langchain_community.graph_vectorstores.extractors.html_link_extractor.HtmlLinkExtractor>`
    - :class:`How to link Documents on common keywords (using KeyBERT) <langchain_community.graph_vectorstores.extractors.keybert_link_extractor.KeybertLinkExtractor>`
    - :class:`How to link Documents on common named entities (using GliNER) <langchain_community.graph_vectorstores.extractors.gliner_link_extractor.GLiNERLinkExtractor>`
    - `langchain-jieba: link extraction tailored for Chinese language <https://github.com/cqzyys/langchain-jieba>`_

Get started
-----------

We chunk the State of the Union text and split it into documents::

    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter

    raw_documents = TextLoader("state_of_the_union.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

Links can be added to documents manually but it's easier to use a
:class:`~langchain_community.graph_vectorstores.extractors.link_extractor.LinkExtractor`.
Several common link extractors are available and you can build your own.
For this guide, we'll use the
:class:`~langchain_community.graph_vectorstores.extractors.keybert_link_extractor.KeybertLinkExtractor`
which uses the KeyBERT model to tag documents with keywords and uses these keywords to
create links between documents::

    from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
    from langchain_community.graph_vectorstores.links import add_links

    extractor = KeybertLinkExtractor()

    for doc in documents:
        add_links(doc, extractor.extract_one(doc))

Create the graph vector store and add documents
-----------------------------------------------

We'll use an Apache Cassandra or Astra DB database as an example.
We create a
:class:`~langchain_community.graph_vectorstores.cassandra.CassandraGraphVectorStore`
from the documents and an :class:`~langchain_openai.embeddings.base.OpenAIEmbeddings`
model::

    import cassio
    from langchain_community.graph_vectorstores import CassandraGraphVectorStore
    from langchain_openai import OpenAIEmbeddings

    # Initialize cassio and the Cassandra session from the environment variables
    cassio.init(auto=True)

    store = CassandraGraphVectorStore.from_documents(
        embedding=OpenAIEmbeddings(),
        documents=documents,
    )


Similarity search
-----------------

If we don't traverse the graph, a graph vector store behaves like a regular vector
store.
So all methods available in a vector store are also available in a graph vector store.
The :meth:`~langchain_community.graph_vectorstores.base.GraphVectorStore.similarity_search`
method returns documents similar to a query without considering
the links between documents::

    docs = store.similarity_search(
        "What did the president say about Ketanji Brown Jackson?"
    )

Traversal search
----------------

The :meth:`~langchain_community.graph_vectorstores.base.GraphVectorStore.traversal_search`
method returns documents similar to a query considering the links
between documents. It first does a similarity search and then traverses the graph to
find linked documents::

    docs = list(
        store.traversal_search("What did the president say about Ketanji Brown Jackson?")
    )

Async methods
-------------

The graph vector store has async versions of the methods prefixed with ``a``::

    docs = [
        doc
        async for doc in store.atraversal_search(
            "What did the president say about Ketanji Brown Jackson?"
        )
    ]

Graph vector store retriever
----------------------------

The graph vector store can be converted to a retriever.
It is similar to the vector store retriever but it also has traversal search methods
such as ``traversal`` and ``mmr_traversal``::

    retriever = store.as_retriever(search_type="mmr_traversal")
    docs = retriever.invoke("What did the president say about Ketanji Brown Jackson?")

"""  # noqa: E501

from langchain_community.graph_vectorstores.base import (
    GraphVectorStore,
    GraphVectorStoreRetriever,
    Node,
)
from langchain_community.graph_vectorstores.cassandra import CassandraGraphVectorStore
from langchain_community.graph_vectorstores.links import (
    Link,
)
from langchain_community.graph_vectorstores.mmr_helper import MmrHelper

__all__ = [
    "GraphVectorStore",
    "GraphVectorStoreRetriever",
    "Node",
    "Link",
    "CassandraGraphVectorStore",
    "MmrHelper",
]
