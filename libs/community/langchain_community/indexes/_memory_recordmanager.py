from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from langchain_community.indexes.base import RecordManager


class MemoryRecordManager(RecordManager):
    data: List[Dict[str, Any]] = []

    def __init__(self, namespace: str):
        super().__init__(namespace=namespace)

    def create_schema(self) -> None:
        pass

    async def acreate_schema(self) -> None:
        pass

    def get_time(self) -> float:
        return datetime.now().timestamp()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        if group_ids is None:
            group_ids = [None] * len(keys)
        if len(keys) != len(group_ids):
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of "
                f"group_ids ({len(group_ids)})"
            )

        update_time = self.get_time()
        if time_at_least and update_time < time_at_least:
            # Safeguard against time sync issues
            raise AssertionError(f"Time sync issue: {update_time} < {time_at_least}")

        records_to_upsert = [
            {
                "key": key,
                "namespace": self.namespace,
                "updated_at": update_time,
                "group_id": group_id,
            }
            for key, group_id in zip(keys, group_ids)
        ]
        self.delete_keys(keys)
        self.data.extend(records_to_upsert)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        return [
            len(list(filter(lambda record: record["key"] == key, self.data))) == 1
            for key in keys
        ]

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        keys = [
            record["key"]
            for record in filter(
                lambda record: record["namespace"] == self.namespace
                and (not group_ids or record["group_id"] in group_ids)
                and (not before or record["updated_at"] < before)
                and (not after or record["updated_at"] > after),
                self.data,
            )
        ]
        return keys[:limit]

    def delete_keys(self, keys: Sequence[str]) -> None:
        self.data = list(
            filter(
                lambda record: record["namespace"] != self.namespace
                or record["key"] not in keys,
                self.data,
            )
        )

    # %% Async versions
    async def aget_time(self) -> float:
        return datetime.now().timestamp()

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        return self.update(keys=keys, group_ids=group_ids, time_at_least=time_at_least)

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        return self.exists(keys=keys)

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

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        return self.delete_keys(keys=keys)
