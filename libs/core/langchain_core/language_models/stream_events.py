"""Shared primitives for provider-native streaming-event converters.

Provider packages implementing `_stream_chat_model_events` import these
to drive raw API events into the same content-block lifecycle the compat
bridge produces, so native and bridged paths emit identical event shapes.
"""

from langchain_core.language_models._compat_bridge import (
    BlockStreamTracker,
    finalize_tool_call_chunk,
)
from langchain_core.language_models._compat_bridge import (
    _accumulate_usage as accumulate_usage,
)
from langchain_core.language_models._compat_bridge import (
    _build_message_finish as build_message_finish,
)
from langchain_core.language_models._compat_bridge import (
    _iter_protocol_blocks as iter_protocol_blocks,
)

__all__ = [
    "BlockStreamTracker",
    "accumulate_usage",
    "build_message_finish",
    "finalize_tool_call_chunk",
    "iter_protocol_blocks",
]
