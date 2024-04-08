from importlib import metadata

from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from langchain_postgres.checkpoint import (
    CheckpointSerializer,
    PickleCheckpointSerializer,
    PostgresCheckpoint,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [
    "__version__",
    "CheckpointSerializer",
    "PostgresChatMessageHistory",
    "PostgresCheckpoint",
    "PickleCheckpointSerializer",
]
