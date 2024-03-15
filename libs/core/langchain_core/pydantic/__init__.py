# type: ignore
from .config import _PYDANTIC_MAJOR_VERSION, _PYDANTIC_VERSION, USE_PYDANTIC_V2

# This is a compatibility layer that pydantic 2 proper, pydantic.v1 in pydantic 2,
# or pydantic 1
try:
    if USE_PYDANTIC_V2:
        from pydantic import BaseModel, Field, PrivateAttr
    else:
        from pydantic.v1 import BaseModel, Field, PrivateAttr
except ImportError:
    from pydantic import BaseModel, Field, PrivateAttr

# Only expose things that are common across all pydantic versions
__all__ = [  # noqa: F405
    "BaseModel",
    "Field",
    "PrivateAttr",
    "_PYDANTIC_MAJOR_VERSION",
    "_PYDANTIC_VERSION",
    "USE_PYDANTIC_V2",
]
