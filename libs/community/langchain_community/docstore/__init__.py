"""**Docstores** are classes to store and load Documents.

The **Docstore** is a simplified version of the Document Loader.

**Class hierarchy:**

.. code-block::

    Docstore --> <name> # Examples: InMemoryDocstore, Wikipedia

**Main helpers:**

.. code-block::

    Document, AddableMixin
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.docstore.arbitrary_fn import (
        DocstoreFn,  # noqa: F401
    )
    from langchain_community.docstore.in_memory import (
        InMemoryDocstore,  # noqa: F401
    )
    from langchain_community.docstore.wikipedia import (
        Wikipedia,  # noqa: F401
    )

__all__ = ["DocstoreFn", "InMemoryDocstore", "Wikipedia"]

_module_lookup = {
    "DocstoreFn": "langchain_community.docstore.arbitrary_fn",
    "InMemoryDocstore": "langchain_community.docstore.in_memory",
    "Wikipedia": "langchain_community.docstore.wikipedia",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
