from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


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
