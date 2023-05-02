Indexes
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/indexing>`_


Indexes refer to ways to structure documents so that LLMs can best interact with them.
This module contains utility functions for working with documents, different types of indexes, and then examples for using those indexes in chains.

The most common way that indexes are used in chains is in a "retrieval" step.
This step refers to taking a user's query and returning the most relevant documents.
We draw this distinction because (1) an index can be used for other things besides retrieval, and (2) retrieval can use other logic besides an index to find relevant documents.
We therefore have a concept of a "Retriever" interface - this is the interface that most chains work with.

Most of the time when we talk about indexes and retrieval we are talking about indexing and retrieving unstructured data (like text documents).
For interacting with structured data (SQL tables, etc) or APIs, please see the corresponding use case sections for links to relevant functionality.
The primary index and retrieval types supported by LangChain are currently centered around vector databases, and therefore
a lot of the functionality we dive deep on those topics.

For an overview of everything related to this, please see the below notebook for getting started:

.. toctree::
   :maxdepth: 1

   ./indexes/getting_started.ipynb

We then provide a deep dive on the four main components.

**Document Loaders**

How to load documents from a variety of sources.

**Text Splitters**

An overview of the abstractions and implementions around splitting text.


**VectorStores**

An overview of VectorStores and the many integrations LangChain provides.


**Retrievers**

An overview of Retrievers and the implementations LangChain provides.

Go Deeper
---------


.. toctree::
   :maxdepth: 1

   ./indexes/document_loaders.rst
   ./indexes/text_splitters.rst
   ./indexes/vectorstores.rst
   ./indexes/retrievers.rst

