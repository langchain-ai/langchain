"""Module defines an index manger interface.

This index manager will provide a control plane for existing indexes.

It also provides a path toward providing implementations the ability to
take care of schema migrations!
"""

import abc
from typing import Any, List, TypedDict


class CreateResponse(TypedDict):
    """Response to a request to create the index."""

    id: str


class IndexInformation(TypedDict):
    """Information about an index."""

    id: str
    """A string that uniquely identifies the index. Required."""


class Index(abc.ABC):
    """Our abstraction (not implemented here.)

    Vectorstores wll sub-class from this to provide their own version.
    """


class IndexManager(abc.ABC):
    @abc.abstractmethod
    def list_indexes(self, **kwargs: Any) -> List[IndexInformation]:
        """List all indexes.

        Implementations are free to parameterize this method with any
        additional arguments they see fit (e.g., pagination, filtering,
        etc.)
        """

    @abc.abstractmethod
    def create_index(self, **kwargs) -> str:
        """Create an index."""

    @abc.abstractmethod
    def get_index(self, id: str) -> Index:
        """Initialize an index by its id."""

    @abc.abstractmethod
    def delete_index(self, id: str) -> Any:
        """Delete an index."""
