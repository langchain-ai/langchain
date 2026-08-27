import os
import tempfile
import time
from collections import defaultdict
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.memory import InMemorySaver, PersistentDict
from langgraph.checkpoint.serde.base import (
    SerializerProtocol,
)
from langgraph.pregel._checkpoint import copy_checkpoint


class MemorySaverAssertImmutable(InMemorySaver):
    storage_for_copies: defaultdict[str, dict[str, dict[str, tuple[str, bytes]]]]

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
        put_sleep: float | None = None,
    ) -> None:
        _, filename = tempfile.mkstemp()

        class TempfilePersistentDict(PersistentDict):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, filename=filename, **kwargs)

        super().__init__(serde=serde, factory=TempfilePersistentDict)
        self.storage_for_copies = defaultdict(lambda: defaultdict(dict))
        self.put_sleep = put_sleep
        self.stack.callback(os.remove, filename)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if self.put_sleep:
            time.sleep(self.put_sleep)
        # assert checkpoint hasn't been modified since last written
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        if saved := super().get(config):
            assert (
                self.serde.loads_typed(
                    self.storage_for_copies[thread_id][checkpoint_ns][saved["id"]]
                )
                == saved
            )
        self.storage_for_copies[thread_id][checkpoint_ns][checkpoint["id"]] = (
            self.serde.dumps_typed(copy_checkpoint(checkpoint))
        )
        # call super to write checkpoint
        return super().put(config, checkpoint, metadata, new_versions)
