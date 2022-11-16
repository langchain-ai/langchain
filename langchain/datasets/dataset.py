from abc import ABC, abstractmethod

import os
from example import Example

LANGCHAIN_STOP_SEQUENCE = " <LCEND>\n"
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache/langchain")


class Dataset(ABC):
    def __init__(self, stop_sequence: str = LANGCHAIN_STOP_SEQUENCE):
        self.stop_sequence = stop_sequence

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Example:
        raise NotImplementedError

    def _maybe_load_cached_data(self, filename: str) -> str:
        """Load cached data from disk if it exists, otherwise return None."""
        filename = f"{CACHE_PATH}/{filename}"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return f.read()
        return None

    def _maybe_cache_data(self, filename: str, data: str) -> None:
        """Save data to disk if it doesn't already exist."""
        filename = f"{CACHE_PATH}/{filename}"
        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(data)
