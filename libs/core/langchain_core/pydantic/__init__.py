# type: ignore
from .config import _PYDANTIC_MAJOR_VERSION, _PYDANTIC_VERSION, USE_PYDANTIC_V2

# This is a compatibility layer that pydantic 2 proper, pydantic.v1 in pydantic 2,
# or pydantic 1
try:
    if USE_PYDANTIC_V2:
        from pydantic import BaseModel, Field  # noqa: F403 # type: ignore
    else:
        from pydantic.v1 import BaseModel, Field  # noqa: F403 # type: ignore
except ImportError:
    from pydantic import BaseModel, Field  # noqa: F403 # type: ignore

# Only expose things that are common across all pydantic versions
__all__ = [  # noqa: F405
    "USE_PYDANTIC_V2",
    "BaseModel",
    "Field",
    "_PYDANTIC_VERSION",
    "_PYDANTIC_MAJOR_VERSION",
]
