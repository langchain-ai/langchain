"""Helper functions for managing the LangChain API.

This module is only relevant for LangChain developers, not for users.

.. warning::

    This module and its submodules are for internal use only.  Do not use them
    in your own code.  We may change the API at any time with no warning.

"""

from .deprecation import (
    LangChainDeprecationWarning,
    deprecated,
    suppress_langchain_deprecation_warning,
)

__all__ = [
    "deprecated",
    "LangChainDeprecationWarning",
    "suppress_langchain_deprecation_warning",
]
