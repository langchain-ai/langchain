from .config import _PYDANTIC_MAJOR_VERSION, _PYDANTIC_VERSION, USE_PYDANTIC_V2

## Create namespaces for pydantic v1 and v2.
# This code must stay at the top of the file before other modules may
# attempt to import pydantic since it adds pydantic_v1 and pydantic_v2 to sys.modules.
#
# This hack is done for the following reasons:
# * Langchain will attempt to remain compatible with both pydantic v1 and v2 since
#   both dependencies and dependents may be stuck on either version of v1 or v2.
# * Creating namespaces for pydantic v1 and v2 should allow us to write code that
#   unambiguously uses either v1 or v2 API.
# * This change is easier to roll out and roll back.


try:
    if USE_PYDANTIC_V2:
        from pydantic import *  # noqa: F403 # type: ignore
    else:
        from pydantic.v1 import *  # noqa: F403 # type: ignore
except ImportError:
    from pydantic import *  # noqa: F403 # type: ignore

# Only expose things that are common across all pydantic versions
__all__ = [
    "BaseModel",  # noqa: F405
    "Field",  # noqa: F405
    "PrivateAttr",  # noqa: F405
    "SecretStr",  # noqa: F405
    "ValidationError",  # noqa: F405
    "USE_PYDANTIC_V2",
    "_PYDANTIC_VERSION",
    "_PYDANTIC_MAJOR_VERSION",
]
