"""Simple in memory docstore in the form of a dict."""
from typing import Dict, Union

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document


class InMemoryDocstore(Docstore, AddableMixin):
    """Simple in memory docstore in the form of a dict."""

    def __init__(self, _dict: Dict[str, Document]):
        """Initialize with dict."""
        self._dict = _dict

    def add(self, texts: Dict[str, Document]) -> None:
        """Add texts to in memory dictionary."""
        overlapping = set(texts).intersection(self._dict)
        if overlapping:
            raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        self._dict = dict(self._dict, **texts)

    def search(self, search: str) -> Union[str, Document]:
        """Search via direct lookup."""
        if search not in self._dict:
            return f"ID {search} not found."
        else:
            return self._dict[search]
