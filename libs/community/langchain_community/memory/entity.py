import logging
from itertools import islice
from typing import Any, Iterable, Optional

from langchain_core.memory import BaseEntityStore

from langchain_community.utilities.redis import get_client

logger = logging.getLogger(__name__)


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


class RedisEntityStore(BaseEntityStore):
    """Redis-backed Entity store.

    Entities get a TTL of 1 day by default, and
    that TTL is extended by 3 days every time the entity is read back.
    """

    redis_client: Any
    session_id: str = "default"
    key_prefix: str = "memory_store"
    ttl: Optional[int] = 60 * 60 * 24
    recall_ttl: Optional[int] = 60 * 60 * 24 * 3

    def __init__(
        self,
        session_id: str = "default",
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "memory_store",
        ttl: Optional[int] = 60 * 60 * 24,
        recall_ttl: Optional[int] = 60 * 60 * 24 * 3,
        *args: Any,
        **kwargs: Any,
    ):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        super().__init__(*args, **kwargs)

        try:
            self.redis_client = get_client(redis_url=url, decode_responses=True)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

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
        logger.debug(f"REDIS MEM get '{self.full_key_prefix}:{key}': '{res}'")
        return res

    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)
        self.redis_client.set(f"{self.full_key_prefix}:{key}", value, ex=self.ttl)
        logger.debug(
            f"REDIS MEM set '{self.full_key_prefix}:{key}': '{value}' EX {self.ttl}"
        )

    def delete(self, key: str) -> None:
        self.redis_client.delete(f"{self.full_key_prefix}:{key}")

    def exists(self, key: str) -> bool:
        return self.redis_client.exists(f"{self.full_key_prefix}:{key}") == 1

    def clear(self) -> None:
        # iterate a list in batches of size batch_size
        def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[Any]:
            iterator = iter(iterable)
            while batch := list(islice(iterator, batch_size)):
                yield batch

        for keybatch in batched(
            self.redis_client.scan_iter(f"{self.full_key_prefix}:*"), 500
        ):
            self.redis_client.delete(*keybatch)


class SQLiteEntityStore(BaseEntityStore):
    """SQLite-backed Entity store"""

    session_id: str = "default"
    table_name: str = "memory_store"
    conn: Any = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        session_id: str = "default",
        db_file: str = "entities.db",
        table_name: str = "memory_store",
        *args: Any,
        **kwargs: Any,
    ):
        try:
            import sqlite3
        except ImportError:
            raise ImportError(
                "Could not import sqlite3 python package. "
                "Please install it with `pip install sqlite3`."
            )
        super().__init__(*args, **kwargs)

        self.conn = sqlite3.connect(db_file)
        self.session_id = session_id
        self.table_name = table_name
        self._create_table_if_not_exists()

    @property
    def full_table_name(self) -> str:
        return f"{self.table_name}_{self.session_id}"

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.full_table_name} (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        with self.conn:
            self.conn.execute(create_table_query)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        query = f"""
            SELECT value
            FROM {self.full_table_name}
            WHERE key = ?
        """
        cursor = self.conn.execute(query, (key,))
        result = cursor.fetchone()
        if result is not None:
            value = result[0]
            return value
        return default

    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)
        query = f"""
            INSERT OR REPLACE INTO {self.full_table_name} (key, value)
            VALUES (?, ?)
        """
        with self.conn:
            self.conn.execute(query, (key, value))

    def delete(self, key: str) -> None:
        query = f"""
            DELETE FROM {self.full_table_name}
            WHERE key = ?
        """
        with self.conn:
            self.conn.execute(query, (key,))

    def exists(self, key: str) -> bool:
        query = f"""
            SELECT 1
            FROM {self.full_table_name}
            WHERE key = ?
            LIMIT 1
        """
        cursor = self.conn.execute(query, (key,))
        result = cursor.fetchone()
        return result is not None

    def clear(self) -> None:
        query = f"""
            DELETE FROM {self.full_table_name}
        """
        with self.conn:
            self.conn.execute(query)
