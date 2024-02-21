import torch

from typing import Dict, Optional, TypeVar

from text_generation_server.models.types import Batch

B = TypeVar("B", bound=Batch)


class Cache:
    def __init__(self):
        self.cache: Dict[int, B] = {}

    def pop(self, batch_id: int) -> Optional[B]:
        return self.cache.pop(batch_id, None)

    def set(self, entry: B):
        if entry is not None:
            self.cache[entry.batch_id] = entry

    def delete(self, batch_id: int):
        batch = self.pop(batch_id)
        if batch is not None:
            del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self):
        keys = list(self.cache.keys())
        for k in keys:
            self.delete(k)

    def __len__(self):
        return len(self.cache.keys())
