Retrievers
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/indexing/retriever>`_


The retriever interface is a generic interface that makes it easy to combine documents with
language models. This interface exposes a `get_relevant_documents` method which takes in a query
(a string) and returns a list of documents.

Please see below for a list of all the retrievers supported.


.. toctree::
   :maxdepth: 1
   :glob:

   ./retrievers/examples/*