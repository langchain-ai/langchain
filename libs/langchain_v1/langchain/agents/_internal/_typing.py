from __future__ import annotations

from collections.abc import Awaitable
from typing import Callable, TypeVar, Union

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

SyncOrAsync = Callable[P, Union[R, Awaitable[R]]]
