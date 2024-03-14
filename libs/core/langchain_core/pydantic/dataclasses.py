from .config import USE_PYDANTIC_V2

try:
    if USE_PYDANTIC_V2:
        from pydantic import *  # noqa: F403
    else:
        from pydantic.v1.dataclasses import *  # noqa: F403
except ImportError:
    from pydantic.dataclasses import *  # noqa: F403
