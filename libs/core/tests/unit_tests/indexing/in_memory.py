import time
from typing import Dict, List, Optional, Sequence, TypedDict

from langchain_core.indexing.base import RecordManager


class _Record(TypedDict):
    group_id: Optional[str]
    updated_at: float


class InMemoryRecordManager(RecordManager):
    """An in-memory record manager for testing purposes."""

    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        # Each key points to a dictionary
        # of {'group_id': group_id, 'updated_at': timestamp}
        self.records: Dict[str, _Record] = {}
        self.namespace = namespace

    def create_schema(self) -> None:
        """In-memory schema creation is simply ensuring the structure is initialized."""

    async def acreate_schema(self) -> None:
        """In-memory schema creation is simply ensuring the structure is initialized."""

    def get_time(self) -> float:
        """Get the current server time as a high resolution timestamp!"""
        return time.time()

    async def aget_time(self) -> float:
        """Get the current server time as a high resolution timestamp!"""
        return self.get_time()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        if group_ids and len(keys) != len(group_ids):
            raise ValueError("Length of keys must match length of group_ids")
        for index, key in enumerate(keys):
            group_id = group_ids[index] if group_ids else None
            if time_at_least and time_at_least > self.get_time():
                raise ValueError("time_at_least must be in the past")
            self.records[key] = {"group_id": group_id, "updated_at": self.get_time()}

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        return [key in self.records for key in keys]

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        return self.exists(keys)

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        result = []
        for key, data in self.records.items():
            if before and data["updated_at"] >= before:
                continue
            if after and data["updated_at"] <= after:
                continue
            if group_ids and data["group_id"] not in group_ids:
                continue
            result.append(key)
        if limit:
            return result[:limit]
        return result

    async def alist_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        return self.list_keys(
            before=before, after=after, group_ids=group_ids, limit=limit
        )

    def delete_keys(self, keys: Sequence[str]) -> None:
        for key in keys:
            if key in self.records:
                del self.records[key]

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        self.delete_keys(keys)
