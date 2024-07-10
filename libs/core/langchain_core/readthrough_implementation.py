from typing import Sequence, List, Optional, Tuple

from langchain_core.documents.base import Blob as MediaBlob
from langchain_core.stores import BaseStore, K, V


class FileAPI(BaseStore[str, MediaBlob]):
    def __init__(self):
        self.client = client

    def mget(self, keys: Sequence[str]) -> List[Optional[MediaBlob]]:
        assert len(keys) == 1
        key = keys[0]
        url = "url_from_file_api/{}".format(key)
        if self.client.exists(key):
            return [
                MediaBlob(
                    path=url,
                )
            ]
        else:
            return [None]

    def mset(self, items: Sequence[Tuple[str, MediaBlob]]) -> MediaBlob:
        assert len(items) == 1
        key, value = items[0]
        url = "url_from_file_api/{}".format(key)
        self.client.upload(url, value)
        return MediaBlob(
            path=url,
        )


class CloudStorage(BaseStore[str, MediaBlob]):
    def __init__(self):
        self.cloud_client = ...

    def mget(self, keys: Sequence[str]) -> List[Optional[MediaBlob]]:
        assert len(keys) == 1
        url = "some_other_url_url_from_file_api/{}".format(key)
        content = self.client.get(url)
        if content:
            return [
                MediaBlob(
                    data=content,  # <-- Can return data and path?
                    path=url,
                )
            ]
        else:
            return [None]

    def mset(self, items: Sequence[Tuple[str, MediaBlob]]) -> MediaBlob:
        url = "url_from_file_api/{}".format(key)
        self.client.upload(url, value)


class ReadthroughStore(BaseStore[str, MediaBlob]):
    def __init__(self, store: BaseStore[str, MediaBlob], backing_store: BaseStore[str, MediaBlob]) -> None:
        self.store = store
        self.backing_store = backing_store

    def mget(self, keys: Sequence[str]) -> List[Optional[MediaBlob]]:
        assert len(keys) == 1
        key = keys[0]
        result = self.store.mget([key])
        if result[0] is not None:
            return result
        else:
            value = self.backing_store.mget([key])
            if value[0] is not None:
                new_value = self.store.mset([(key, value[0])])
                return [new_value]
        return [None]

    def mset(self, items: Sequence[Tuple[str, MediaBlob]]) -> MediaBlob:
        # Readthrough cache does not allow setting value
        raise NotImplementedError()


[in_memory_store, file_api_store, cloud_store, web_store] =>
ReadthroughStore(
    in_memory_store,
    ReadthroughStore(
        file_api_store,
        ReadthroughStore(
            cloud_store,
            web_store
        )
    )
)


class LayeredReadthrough(BaseStore[str, MediaBlob]):
    def __init__(self, stores: Sequence[BaseStore[str, MediaBlob]]) -> None:
        reversed_stores = stores[::-1]
        backing_store = reversed_stores[0]

        readthrough_store = ReadthroughStore(reversed_stores[1], backing_store)

        for store in reversed_stores[2:]:
            readthrough_store = ReadthroughStore(store, readthrough_store)

        self.readthrough_store = readthrough_store

    def mget(self, keys: Sequence[K]) -> List[Optional[V]]:
        return self.readthrough_store.mget(keys)


----

MediaManager = {
    Hasher
    LayeredReadthrough([in_memory_lru_store, file_api_store, cloud_store, web_store])
}
