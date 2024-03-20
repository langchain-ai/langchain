from __future__ import annotations

import logging
from itertools import islice
from typing import TYPE_CHECKING, Any, Iterable, Optional

from langchain_core.entity_stores import BaseEntityStore

from langchain_community.utilities.redis import _redis_sentinel_client

if TYPE_CHECKING:
    from redis import Redis as RedisType

logger = logging.getLogger(__name__)


class RedisEntityStore(BaseEntityStore):
    """Redis-backed Entity store.

    Entities get a TTL of 1 day by default, and
    that TTL is extended by 3 days every time the entity is read back.

    Must have `redis` and `langchain-community` installed.
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
            self.redis_client = self._get_client(redis_url=url, decode_responses=True)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.recall_ttl = recall_ttl or ttl

    @staticmethod
    def _get_client(redis_url: str, **kwargs: Any) -> RedisType:
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis>=4.1.0`."
            )

        # check if normal redis:// or redis+sentinel:// url
        if redis_url.startswith("redis+sentinel"):
            redis_client = _redis_sentinel_client(redis_url, **kwargs)
        elif redis_url.startswith(
            "rediss+sentinel"
        ):  # sentinel with TLS support enables
            kwargs["ssl"] = True
            if "ssl_cert_reqs" not in kwargs:
                kwargs["ssl_cert_reqs"] = "none"
            redis_client = _redis_sentinel_client(redis_url, **kwargs)
        else:
            # connect to redis server from url, reconnect with cluster client if needed
            redis_client = redis.from_url(redis_url, **kwargs)

            try:
                cluster_info = redis_client.info("cluster")
                cluster_enabled = cluster_info["cluster_enabled"] == 1
            except redis.exceptions.RedisError:
                cluster_enabled = False
            if cluster_enabled:
                from redis.cluster import RedisCluster

                redis_client.close()
                return RedisCluster.from_url(redis_url, **kwargs)
        return redis_client

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
