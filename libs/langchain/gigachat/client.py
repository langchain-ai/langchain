import logging
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    TypeVar,
    Union,
)

import httpx

from .api import post_chat, post_oauth, post_token, stream_chat
from .exceptions import AuthenticationError
from .models import AccessToken, Chat, ChatCompletion, ChatCompletionChunk, Token
from .settings import Settings

T = TypeVar("T")

logger = logging.getLogger(__name__)


def _get_kwargs(settings: Settings) -> Dict[str, Any]:
    """Настройки для подключения к API GIGACHAT"""
    return {
        "base_url": settings.api_base_url,
        "verify": settings.verify_ssl,
        "timeout": httpx.Timeout(settings.timeout),
    }


def _get_oauth_kwargs(settings: Settings) -> Dict[str, Any]:
    """Настройки для подключения к серверу авторизации OAuth 2.0"""
    return {
        "base_url": settings.oauth_base_url,
        "verify": settings.oauth_verify_ssl,
        "timeout": httpx.Timeout(settings.oauth_timeout),
    }


def _parse_chat(chat: Union[Chat, Dict[str, Any]], model: Optional[str]) -> Chat:
    payload = Chat.parse_obj(chat)
    if model:
        payload.model = model
    return payload


def _build_access_token(token: Token) -> AccessToken:
    return AccessToken(access_token=token.tok, expires_at=token.exp)


class _BaseClient:
    _access_token: Optional[AccessToken] = None

    def __init__(
        self,
        *,
        verbose: Optional[bool] = None,
        use_auth: Optional[bool] = None,
        api_base_url: Optional[str] = None,
        token: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        verify_ssl: Optional[bool] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        oauth_base_url: Optional[str] = None,
        oauth_token: Optional[str] = None,
        oauth_scope: Optional[str] = None,
        oauth_timeout: Optional[float] = None,
        oauth_verify_ssl: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        config = {
            key: value
            for key, value in locals().items()
            if key not in ("self", "kwargs") and value is not None
        }
        self._settings = Settings(**config)
        if self._settings.token:
            self._access_token = AccessToken(
                access_token=self._settings.token, expires_at=0
            )

    @property
    def token(self) -> Optional[str]:
        if self._settings.use_auth:
            if self._access_token:
                return self._access_token.access_token
        return None

    @property
    def _use_auth(self) -> bool:
        return self._settings.use_auth

    def _check_validity_token(self) -> bool:
        """Проверить время завершения действия токена"""
        if self._access_token:
            # _check_validity_token
            return True
        return False

    def _reset_token(self) -> None:
        """Сбросить токен"""
        self._access_token = None


class GigaChatSyncClient(_BaseClient):
    """Синхронный клиент GigaChat"""

    @cached_property
    def _client(self) -> httpx.Client:
        return httpx.Client(**_get_kwargs(self._settings))

    @cached_property
    def _oauth_client(self) -> httpx.Client:
        return httpx.Client(**_get_oauth_kwargs(self._settings))

    def close(self) -> None:
        self._client.close()
        self._oauth_client.close()

    def __enter__(self) -> "GigaChatSyncClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _update_token(self) -> None:
        if self._settings.user and self._settings.password:
            self._access_token = _build_access_token(
                post_token.sync(
                    self._client, self._settings.user, self._settings.password
                )
            )
            logger.info("UPDATE TOKEN")
        elif self._settings.oauth_token:
            self._access_token = post_oauth.sync(
                self._oauth_client,
                self._settings.oauth_token,
                self._settings.oauth_scope,
            )
            logger.info("OAUTH UPDATE TOKEN")
        else:
            logger.info("IGNORE UPDATE TOKEN")

    def chat(self, payload: Union[Chat, Dict[str, Any]]) -> ChatCompletion:
        chat = _parse_chat(payload, model=self._settings.model)

        if self._use_auth:
            if self._check_validity_token():
                try:
                    return post_chat.sync(self._client, chat, self.token)
                except AuthenticationError:
                    logger.warning("AUTHENTICATION ERROR")
                    self._reset_token()
            self._update_token()

        return post_chat.sync(self._client, chat, self.token)

    def stream(
        self, payload: Union[Chat, Dict[str, Any]]
    ) -> Iterator[ChatCompletionChunk]:
        chat = _parse_chat(payload, model=self._settings.model)

        if self._use_auth:
            if self._check_validity_token():
                try:
                    for chunk in stream_chat.sync(self._client, chat, self.token):
                        yield chunk
                    return
                except AuthenticationError:
                    logger.warning("AUTHENTICATION ERROR")
                    self._reset_token()
            self._update_token()

        for chunk in stream_chat.sync(self._client, chat, self.token):
            yield chunk


class GigaChatAsyncClient(_BaseClient):
    """Асинхронный клиент GigaChat"""

    @cached_property
    def _aclient(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(**_get_kwargs(self._settings))

    @cached_property
    def _oauth_aclient(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(**_get_oauth_kwargs(self._settings))

    async def aclose(self) -> None:
        await self._aclient.aclose()
        await self._oauth_aclient.aclose()

    async def __aenter__(self) -> "GigaChatAsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _aupdate_token(self) -> None:
        if self._settings.user and self._settings.password:
            self._access_token = _build_access_token(
                await post_token.asyncio(
                    self._aclient, self._settings.user, self._settings.password
                )
            )
            logger.info("UPDATE TOKEN")
        elif self._settings.oauth_token:
            self._access_token = await post_oauth.asyncio(
                self._oauth_aclient,
                self._settings.oauth_token,
                self._settings.oauth_scope,
            )
            logger.info("OAUTH UPDATE TOKEN")
        else:
            logger.info("IGNORE UPDATE TOKEN")

    async def achat(self, payload: Union[Chat, Dict[str, Any]]) -> ChatCompletion:
        chat = _parse_chat(payload, model=self._settings.model)

        if self._use_auth:
            if self._check_validity_token():
                try:
                    return await post_chat.asyncio(self._aclient, chat, self.token)
                except AuthenticationError:
                    logger.warning("AUTHENTICATION ERROR")
                    self._reset_token()
            await self._aupdate_token()

        return await post_chat.asyncio(self._aclient, chat, self.token)

    async def astream(
        self, payload: Union[Chat, Dict[str, Any]]
    ) -> AsyncIterator[ChatCompletionChunk]:
        chat = _parse_chat(payload, model=self._settings.model)

        if self._use_auth:
            if self._check_validity_token():
                try:
                    async for chunk in stream_chat.asyncio(
                        self._aclient, chat, self.token
                    ):
                        yield chunk
                    return
                except AuthenticationError:
                    logger.warning("AUTHENTICATION ERROR")
                    self._reset_token()
            await self._aupdate_token()

        async for chunk in stream_chat.asyncio(self._aclient, chat, self.token):
            yield chunk
