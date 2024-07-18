from importlib import metadata
from typing import Any, Type

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
    from pydantic.v1 import *  # noqa: F403
except ImportError:
    from pydantic import *  # type: ignore # noqa: F403


try:
    _PYDANTIC_MAJOR_VERSION: int = int(metadata.version("pydantic").split(".")[0])
except metadata.PackageNotFoundError:
    _PYDANTIC_MAJOR_VERSION = 0


def _issubclass_base_model(type_: Type) -> bool:
    from pydantic import BaseModel

    if _PYDANTIC_MAJOR_VERSION == 2:
        from pydantic.v1 import BaseModel as BaseModelV1

        return issubclass(type_, (BaseModel, BaseModelV1))
    else:
        return issubclass(type_, BaseModel)


def _isinstance_base_model(obj: Any) -> bool:
    from pydantic import BaseModel

    if _PYDANTIC_MAJOR_VERSION == 2:
        from pydantic.v1 import BaseModel as BaseModelV1

        return isinstance(obj, (BaseModel, BaseModelV1))
    else:
        return isinstance(obj, BaseModel)
