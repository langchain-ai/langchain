from abc import ABC, abstractmethod

class Database(ABC):

    @property
    @abstractmethod
    def dialect(self):
        """Return the dialect description to use in the prompt."""

    @property
    @abstractmethod
    def table_info(self):
        """Return information about the tables available."""

    @abstractmethod
    def run(self, command: str) -> str:
        """Run command and return string of the results."""
