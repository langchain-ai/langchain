from typing import Union, Dict, Any

from langchain.docstore.base import Docstore
from langchain.docstore.document import Document


class InMemoryDocstore(Docstore):

    def __init__(self, _dict: Dict[str, Document]):
        self._dict = _dict
    def search(self, search: str) -> Union[str, Document]:
        if search not in self._dict:
            return f"ID {search} not found."
        else:
            return self._dict[search]