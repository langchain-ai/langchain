"""**Docstores** are classes to store and load Documents.

The **Docstore** is a simplified version of the Document Loader.

**Class hierarchy:**

.. code-block::

    Docstore --> <name> # Examples: InMemoryDocstore, Wikipedia

**Main helpers:**

.. code-block::

    Document, AddableMixin
"""
from langchain.docstore._kw_storage import create_kw_docstore
from langchain.docstore.arbitrary_fn import DocstoreFn
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.wikipedia import Wikipedia

__all__ = ["DocstoreFn", "InMemoryDocstore", "Wikipedia", "create_kw_docstore"]
