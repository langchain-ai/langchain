"""Pydantic v1 compatibility shim."""

from langchain_core._api import warn_deprecated

try:
    from pydantic.v1.dataclasses import *  # noqa: F403
except ImportError:
    from pydantic.dataclasses import *  # type: ignore # noqa: F403

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
