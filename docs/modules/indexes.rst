Indexes
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/indexing>`_


**Indexes** refer to ways to structure documents so that LLMs can best interact with them.

The most common way that indexes are used in chains is in a "retrieval" step.
This step refers to taking a user's query and returning the most relevant documents.
We draw this distinction because (1) an index can be used for other things besides retrieval, and
(2) retrieval can use other logic besides an index to find relevant documents.
We therefore have a concept of a **Retriever** interface - this is the interface that most chains work with.

Most of the time when we talk about indexes and retrieval we are talking about indexing and retrieving
unstructured data (like text documents).
For interacting with structured data (SQL tables, etc) or APIs, please see the corresponding use case
sections for links to relevant functionality.

|
- `Getting Started <./indexes/getting_started.html>`_: An overview of the indexes.


Index Types
---------------------

- `Document Loaders <./indexes/document_loaders.html>`_: How to load documents from a variety of sources.

- `Text Splitters <./indexes/text_splitters.html>`_: An overview and different types of the **Text Splitters**.

- `VectorStores <./indexes/vectorstores.html>`_: An overview and different types of the  **Vector Stores**.

- `Retrievers <./indexes/retrievers.html>`_: An overview and different types of the **Retrievers**.



.. toctree::
   :maxdepth: 1
   :hidden:

   ./indexes/getting_started.ipynb
   ./indexes/document_loaders.rst
   ./indexes/text_splitters.rst
   ./indexes/vectorstores.rst
   ./indexes/retrievers.rst

