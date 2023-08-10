from __future__ import annotations

from typing import Optional

from langchain.load.serializable import Serializable
from langchain.schema.runnable.base import Input, Runnable, RunnableConfig


class RunnablePassthrough(Serializable, Runnable[Input, Input]):
    """
    A runnable that passes through the input.
    """

    @property
    def lc_serializable(self) -> bool:
        return True

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Input:
        return self._call_with_config(lambda x: x, input, config)
