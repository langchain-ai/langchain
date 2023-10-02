from .client import GigaChatAsyncClient, GigaChatSyncClient


class GigaChat(GigaChatSyncClient, GigaChatAsyncClient):
    ...
