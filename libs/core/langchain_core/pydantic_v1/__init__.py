from importlib import metadata

from langchain_core._api.deprecation import warn_deprecated

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

warn_deprecated(
    "0.3.0",
    removal="1.0.0",
    alternative="pydantic.v1 or pydantic",
    message=(
        "As of langchain-core 0.3.0, LangChain uses pydantic v2 internally. "
        "The langchain_core.pydantic_v1 module was a "
        "compatibility shim for pydantic v1, and should no longer be used. "
        "Please update the code to import from Pydantic directly.\n\n"
        "For example, replace imports like: "
        "`from langchain_core.pydantic_v1 import BaseModel`\n"
        "with: `from pydantic import BaseModel`\n"
        "or the v1 compatibility namespace if you are working in a code base "
        "that has not been fully upgraded to pydantic 2 yet. "
        "\tfrom pydantic.v1 import BaseModel\n"
    ),
)
