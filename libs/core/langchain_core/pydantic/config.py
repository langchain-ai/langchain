import os
from importlib import metadata

_PYDANTIC_VERSION = metadata.version("pydantic")

try:
    _PYDANTIC_MAJOR_VERSION: int = int(_PYDANTIC_VERSION.split(".")[0])
except metadata.PackageNotFoundError:
    _PYDANTIC_MAJOR_VERSION = 0


def _get_use_pydantic_v2() -> bool:
    """Get the value of the LC_PYDANTIC_V2_EXPERIMENTAL environment variable."""
    value = os.environ.get("LC_PYDANTIC_V2_EXPERIMENTAL", "false").lower()
    if value == "true":
        if _PYDANTIC_MAJOR_VERSION != 2:
            raise ValueError(
                f"LC_PYDANTIC_V2_EXPERIMENTAL is set to true, "
                f"but pydantic version is {_PYDANTIC_VERSION}"
            )
        return True
    elif value == "false":
        return False
    else:
        raise ValueError(
            f"Invalid value for LANGCHAIN_PYDANTIC_V2_EXPERIMENTAL: {value}"
        )


USE_PYDANTIC_V2 = _get_use_pydantic_v2()
