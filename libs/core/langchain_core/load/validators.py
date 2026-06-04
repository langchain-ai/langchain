"""Init validators for deserialization security.

Each validator is a callable matching the `InitValidator` protocol: it takes a
class path tuple and kwargs dict, returns `None` on success, and raises
`ValueError` if the deserialization should be blocked.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.load.load import InitValidator


CLASS_INIT_VALIDATORS: dict[tuple[str, ...], "InitValidator"] = {}
