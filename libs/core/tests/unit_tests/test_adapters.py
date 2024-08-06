"""Adapters to convert internal messages into dicts for unit-tests."""

from typing import Any, Dict, Union

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


def to_dict(
    obj: Union[Document, BaseMessage], *, drop_id: bool = False
) -> Dict[str, Any]:
    """Convert the object to a dict and potentially drop id"""
    d = obj.dict()
    if drop_id:
        del d["id"]
    return d
