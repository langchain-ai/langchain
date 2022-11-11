"""Simple in memory docstore in the form of a dict."""
from typing import List, Union

from langchain.docstore.base import Docstore
from langchain.docstore.document import Document


class InMemoryDocstore(Docstore):
    """Simple in memory docstore in the form of a dict."""

    def __init__(self, _docs: List[Document]):
        """Initialize with dict."""
        self._dict = {}
        for i, doc in enumerate(_docs):
            self._dict[i] = doc

    def _get_max_idx(self) -> int:
        """Get max index of documents."""
        return max(self._dict.keys())

    def add(self, docs: List[Document]) -> bool:
        """Add more documents."""
        idx = self._get_max_idx() + 1
        for doc in docs:
            self._dict[idx] = doc
            idx += 1
        return True

    def search(self, search: int) -> Union[str, Document]:
        """Search via direct lookup."""
        if search not in self._dict:
            return f"ID {search} not found."
        else:
            return self._dict[search]
