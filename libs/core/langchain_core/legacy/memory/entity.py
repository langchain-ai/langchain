import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

from langchain_core.pydantic_v1 import BaseModel

logger = logging.getLogger(__name__)


class BaseEntityStore(BaseModel, ABC):
    """Abstract base class for Entity store."""

    @abstractmethod
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get entity value from store."""
        pass

    @abstractmethod
    def set(self, key: str, value: Optional[str]) -> None:
        """Set entity value in store."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete entity value from store."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if entity exists in store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all entities from store."""
        pass


class InMemoryEntityStore(BaseEntityStore):
    """In-memory Entity store."""

    store: Dict[str, Optional[str]] = {}

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.store.get(key, default)

    def set(self, key: str, value: Optional[str]) -> None:
        self.store[key] = value

    def delete(self, key: str) -> None:
        del self.store[key]

    def exists(self, key: str) -> bool:
        return key in self.store

    def clear(self) -> None:
        return self.store.clear()
