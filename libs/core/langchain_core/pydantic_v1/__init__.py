from importlib import metadata

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
    from pydantic.v1 import *  # noqa: F403 # type: ignore
except ImportError:
    from pydantic import *  # noqa: F403 # type: ignore


try:
    _PYDANTIC_MAJOR_VERSION: int = int(metadata.version("pydantic").split(".")[0])
except metadata.PackageNotFoundError:
    _PYDANTIC_MAJOR_VERSION = 0
