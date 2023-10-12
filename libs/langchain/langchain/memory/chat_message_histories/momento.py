from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict
from langchain.utils import get_from_env

if TYPE_CHECKING:
    import momento


def _ensure_cache_exists(cache_client: momento.CacheClient, cache_name: str) -> None:
    """Create cache if it doesn't exist.

    Raises:
        SdkException: Momento service or network error
        Exception: Unexpected response
    """
    from momento.responses import CreateCache

    create_cache_response = cache_client.create_cache(cache_name)
    if isinstance(create_cache_response, CreateCache.Success) or isinstance(
        create_cache_response, CreateCache.CacheAlreadyExists
    ):
        return None
    elif isinstance(create_cache_response, CreateCache.Error):
        raise create_cache_response.inner_exception
    else:
        raise Exception(f"Unexpected response cache creation: {create_cache_response}")


class MomentoChatMessageHistory(BaseChatMessageHistory):
    """Chat message history cache that uses Momento as a backend.

    See https://gomomento.com/"""

    def __init__(
        self,
        session_id: str,
        cache_client: momento.CacheClient,
        cache_name: str,
        *,
        key_prefix: str = "message_store:",
        ttl: Optional[timedelta] = None,
        ensure_cache_exists: bool = True,
    ):
        """Instantiate a chat message history cache that uses Momento as a backend.

        Note: to instantiate the cache client passed to MomentoChatMessageHistory,
        you must have a Momento account at https://gomomento.com/.

        Args:
            session_id (str): The session ID to use for this chat session.
            cache_client (CacheClient): The Momento cache client.
            cache_name (str): The name of the cache to use to store the messages.
            key_prefix (str, optional): The prefix to apply to the cache key.
                Defaults to "message_store:".
            ttl (Optional[timedelta], optional): The TTL to use for the messages.
                Defaults to None, ie the default TTL of the cache will be used.
            ensure_cache_exists (bool, optional): Create the cache if it doesn't exist.
                Defaults to True.

        Raises:
            ImportError: Momento python package is not installed.
            TypeError: cache_client is not of type momento.CacheClientObject
        """
        try:
            from momento import CacheClient
            from momento.requests import CollectionTtl
        except ImportError:
            raise ImportError(
                "Could not import momento python package. "
                "Please install it with `pip install momento`."
            )
        if not isinstance(cache_client, CacheClient):
            raise TypeError("cache_client must be a momento.CacheClient object.")
        if ensure_cache_exists:
            _ensure_cache_exists(cache_client, cache_name)
        self.key = key_prefix + session_id
        self.cache_client = cache_client
        self.cache_name = cache_name
        if ttl is not None:
            self.ttl = CollectionTtl.of(ttl)
        else:
            self.ttl = CollectionTtl.from_cache_ttl()

    @classmethod
    def from_client_params(
        cls,
        session_id: str,
        cache_name: str,
        ttl: timedelta,
        *,
        configuration: Optional[momento.config.Configuration] = None,
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,  # for backwards compatibility
        **kwargs: Any,
    ) -> MomentoChatMessageHistory:
        """Construct cache from CacheClient parameters."""
        try:
            from momento import CacheClient, Configurations, CredentialProvider
        except ImportError:
            raise ImportError(
                "Could not import momento python package. "
                "Please install it with `pip install momento`."
            )
        if configuration is None:
            configuration = Configurations.Laptop.v1()

        # Try checking `MOMENTO_AUTH_TOKEN` first for backwards compatibility
        try:
            api_key = auth_token or get_from_env("auth_token", "MOMENTO_AUTH_TOKEN")
        except ValueError:
            api_key = api_key or get_from_env("api_key", "MOMENTO_API_KEY")
        credentials = CredentialProvider.from_string(api_key)
        cache_client = CacheClient(configuration, credentials, default_ttl=ttl)
        return cls(session_id, cache_client, cache_name, ttl=ttl, **kwargs)

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Retrieve the messages from Momento.

        Raises:
            SdkException: Momento service or network error
            Exception: Unexpected response

        Returns:
            list[BaseMessage]: List of cached messages
        """
        from momento.responses import CacheListFetch

        fetch_response = self.cache_client.list_fetch(self.cache_name, self.key)

        if isinstance(fetch_response, CacheListFetch.Hit):
            items = [json.loads(m) for m in fetch_response.value_list_string]
            return messages_from_dict(items)
        elif isinstance(fetch_response, CacheListFetch.Miss):
            return []
        elif isinstance(fetch_response, CacheListFetch.Error):
            raise fetch_response.inner_exception
        else:
            raise Exception(f"Unexpected response: {fetch_response}")

    def add_message(self, message: BaseMessage) -> None:
        """Store a message in the cache.

        Args:
            message (BaseMessage): The message object to store.

        Raises:
            SdkException: Momento service or network error.
            Exception: Unexpected response.
        """
        from momento.responses import CacheListPushBack

        item = json.dumps(_message_to_dict(message))
        push_response = self.cache_client.list_push_back(
            self.cache_name, self.key, item, ttl=self.ttl
        )
        if isinstance(push_response, CacheListPushBack.Success):
            return None
        elif isinstance(push_response, CacheListPushBack.Error):
            raise push_response.inner_exception
        else:
            raise Exception(f"Unexpected response: {push_response}")

    def clear(self) -> None:
        """Remove the session's messages from the cache.

        Raises:
            SdkException: Momento service or network error.
            Exception: Unexpected response.
        """
        from momento.responses import CacheDelete

        delete_response = self.cache_client.delete(self.cache_name, self.key)
        if isinstance(delete_response, CacheDelete.Success):
            return None
        elif isinstance(delete_response, CacheDelete.Error):
            raise delete_response.inner_exception
        else:
            raise Exception(f"Unexpected response: {delete_response}")
