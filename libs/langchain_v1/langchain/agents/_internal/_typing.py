"""Typing utilities for agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

SyncOrAsync = Callable[P, R | Awaitable[R]]
