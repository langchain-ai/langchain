Indexes
==========================

Indexes refer to ways to structure documents so that LLMs can best interact with them.
This module contains utility functions for working with documents, different types of indexes, and then examples for using those indexes in chains.

The most common way that indexes are used in chains is in a "retrieval" step.
This step refers to taking a user's query and returning the most relevant documents.
We draw this distinction because (1) an index can be used for other things besides retrieval, and (2) retrieval can use other logic besides an index to find relevant documents.
We therefor have a concept of a "Retriever" interface - this is the interface that most chains work with.

Most of the time when we talk about indexes and retrieval we are talking about indexing and retrieving unstructured data (like text documents).
For interacting with structured data (SQL tables, etc) or APIs, please see the corresponding use case sections for links to relevant functionality.
The primary index and retrieval types supported by LangChain are currently centered around vector databases, and therefore
a lot of the functionality we dive deep on those topics.

The following sections of documentation are provided:

- `Getting Started <./indexes/getting_started.html>`_: An overview of the base "Retriever" interface, and then all the functionality LangChain provides for working with indexes.

- `Key Concepts <./indexes/key_concepts.html>`_: A conceptual guide going over the various concepts related to indexes and the tools needed to create them.

- `How-To Guides <./indexes/how_to_guides.html>`_: A collection of how-to guides. These highlight how to use all the relevant tools, the different types of vector databases, different types of retrievers, and how to use retrievers and indexes in chains.


.. toctree::
   :maxdepth: 1
   :name: LLMs
   :hidden:

   ./indexes/getting_started.ipynb
   ./indexes/key_concepts.md
   ./indexes/how_to_guides.rst
