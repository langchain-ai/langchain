from __future__ import annotations

from typing import Optional, Any

from langchain.memory.entity import logger
from langchain_core.entity_stores import BaseEntityStore


class UpstashRedisEntityStore(BaseEntityStore):
    """Upstash Redis backed Entity store.

    Entities get a TTL of 1 day by default, and
    that TTL is extended by 3 days every time the entity is read back.
    """

    def __init__(
        self,
        session_id: str = "default",
        url: str = "",
        token: str = "",
        key_prefix: str = "memory_store",
        ttl: Optional[int] = 60 * 60 * 24,
        recall_ttl: Optional[int] = 60 * 60 * 24 * 3,
        *args: Any,
        **kwargs: Any,
    ):
        try:
            from upstash_redis import Redis
        except ImportError:
            raise ImportError(
                "Could not import upstash_redis python package. "
                "Please install it with `pip install upstash_redis`."
            )

        super().__init__(*args, **kwargs)

        try:
            self.redis_client = Redis(url=url, token=token)
        except Exception:
            logger.error("Upstash Redis instance could not be initiated.")

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.recall_ttl = recall_ttl or ttl

    @property
    def full_key_prefix(self) -> str:
        return f"{self.key_prefix}:{self.session_id}"

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        res = (
            self.redis_client.getex(f"{self.full_key_prefix}:{key}", ex=self.recall_ttl)
            or default
            or ""
        )
        logger.debug(f"Upstash Redis MEM get '{self.full_key_prefix}:{key}': '{res}'")
        return res

    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)
        self.redis_client.set(f"{self.full_key_prefix}:{key}", value, ex=self.ttl)
        logger.debug(
            f"Redis MEM set '{self.full_key_prefix}:{key}': '{value}' EX {self.ttl}"
        )

    def delete(self, key: str) -> None:
        self.redis_client.delete(f"{self.full_key_prefix}:{key}")

    def exists(self, key: str) -> bool:
        return self.redis_client.exists(f"{self.full_key_prefix}:{key}") == 1

    def clear(self) -> None:
        def scan_and_delete(cursor: int) -> int:
            cursor, keys_to_delete = self.redis_client.scan(
                cursor, f"{self.full_key_prefix}:*"
            )
            self.redis_client.delete(*keys_to_delete)
            return cursor

        cursor = scan_and_delete(0)
        while cursor != 0:
            scan_and_delete(cursor)
