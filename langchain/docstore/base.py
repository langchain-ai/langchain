"""Interface to access to place that stores documents."""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
)

from langchain.docstore.document import Document

UID_TYPE = TypeVar("UID_TYPE")


class Docstore(ABC):
    """Interface to place that stores documents."""

    @abstractmethod
    def search(self, search: str) -> Union[str, Document]:
        """Search for document.

        If page exists, return the page summary, and a Document object.
        If page does not exist, return similar entries.
        """


class AddableMixin(object, ABC):
    """Mixin class that supports adding texts."""

    @abstractmethod
    def add(self, texts: Dict[str, Document]) -> None:
        """Add more documents."""


class DocManager(ABC, Generic[UID_TYPE]):
    def add(self, doc: Document) -> UID_TYPE:
        if self.contains_doc(doc):
            raise ValueError
        uid = self.generate_uid()
        self._add(doc, uid)
        return uid

    def lazy_add_docs(self, docs: Iterator[Document]) -> Iterator[UID_TYPE]:
        for doc in docs:
            yield self.add(doc)

    def add_docs(self, docs: Sequence[Document]) -> List[UID_TYPE]:
        return list(self.lazy_add_docs(docs))

    def add_text(self, text: str, metadata: Optional[dict] = None) -> UID_TYPE:
        _metadata = metadata or {}
        return self.add(Document(page_content=text, metadata=_metadata))

    def lazy_add_texts(
        self, texts: Iterator[str], metadatas: Optional[Iterator[dict]] = None
    ) -> Iterator[UID_TYPE]:
        _metadatas = metadatas or ({} for _ in texts)
        for text, metadata in zip(texts, _metadatas):
            yield self.add_text(text, metadata=metadata)

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[UID_TYPE]:
        return list(self.lazy_add_texts(texts, metadatas=metadatas))

    @abstractmethod
    def _add(self, doc: Document, uid: UID_TYPE) -> None:
        """"""

    def delete(self, uid: UID_TYPE) -> None:
        """"""
        if not self.get(uid):
            raise ValueError

        return self._delete(uid)

    @abstractmethod
    def _delete(self, uid: UID_TYPE) -> None:
        """"""

    @abstractmethod
    def contains_doc(self, doc: Document) -> bool:
        """"""

    @abstractmethod
    def get_doc_id(self, doc: Document) -> UID_TYPE:
        """"""

    @abstractmethod
    def get(self, uid: UID_TYPE) -> Document:
        """"""

    @abstractmethod
    def generate_uid(self) -> UID_TYPE:
        """"""
        uid = self._generate_uid()
        if self.get(uid):
            raise ValueError
        return uid

    @abstractmethod
    def _generate_uid(self) -> UID_TYPE:
        """"""

    def persist(self, path: Union[str, Path]) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[str, Path]) -> DocManager:
        raise NotImplementedError


class SimpleDocManager(DocManager[str]):
    def __init__(
        self,
        uid_doc_map: Optional[Dict[str, Document]] = None,
        doc_uid_map: Optional[Dict[str, str]] = None,
    ):
        self.uid_doc_map = uid_doc_map or {}
        self.doc_uid_map = doc_uid_map or {}
        if len(self.uid_doc_map) != len(self.doc_uid_map):
            raise ValueError

    def _add(self, doc: Document, uid: str) -> None:
        self.uid_doc_map[uid] = doc
        self.doc_uid_map[self.serialize(doc)] = uid

    def _delete(self, uid: str) -> None:
        doc = self.uid_doc_map[uid]
        del self.doc_uid_map[self.serialize(doc)]
        del self.uid_doc_map[uid]

    def contains_doc(self, doc: Document) -> bool:
        return self.serialize(doc) in self.doc_uid_map

    def get_doc_id(self, doc: Document) -> str:
        return self.doc_uid_map[self.serialize(doc)]

    def serialize(self, doc: Document) -> str:
        # Assumes metadata is JSON-serializable.
        return json.dumps(doc.dict(), sort_keys=True)

    def _generate_uid(self) -> str:
        return str(uuid.uuid4())

    def persist(self, path: Union[str, Path]) -> None:
        uid_doc_map = {
            uid: self.serialize(doc) for uid, doc in self.uid_doc_map.items()
        }
        with open(path, "w") as f:
            json.dump(uid_doc_map, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> SimpleDocManager:
        with open(path, "r") as f:
            serialized_uid_doc_map = json.load(f)
        uid_doc_map = {
            uid: Document(
                page_content=doc_dict["page_content"], metadata=doc_dict["metadata"]
            )
            for uid, doc_dict in serialized_uid_doc_map.items()
        }
        doc_uid_map = {v: k for k, v in serialized_uid_doc_map.items()}
        return cls(uid_doc_map=uid_doc_map, doc_uid_map=doc_uid_map)
