"""**Docstores** are classes to store and load Documents.

The **Docstore** is a simplified version of the Document Loader.

**Class hierarchy:**

.. code-block::

    Docstore --> <name> # Examples: InMemoryDocstore, Wikipedia

**Main helpers:**

.. code-block::

    Document, AddableMixin
"""
from langchain_community.docstore.arbitrary_fn import DocstoreFn
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.docstore.wikipedia import Wikipedia

__all__ = ["DocstoreFn", "InMemoryDocstore", "Wikipedia"]
