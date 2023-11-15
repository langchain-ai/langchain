from pathlib import Path
from typing import Any

from langchain._api.path import as_import_path


def __getattr__(name: str) -> Any:
    """Get attr name."""

    here = as_import_path(Path(__file__).parent)

    old_path = "langchain." + here + "." + name
    new_path = "langchain_experimental." + here + "." + name
    raise AttributeError(
        "This agent has been moved to langchain experiment. "
        "This agent relies on python REPL tool under the hood, so to use it "
        "safely please sandbox the python REPL. "
        "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
        "and https://github.com/langchain-ai/langchain/discussions/11680"
        "To keep using this code as is, install langchain experimental and "
        f"update your import statement from:\n `{old_path}` to `{new_path}`."
    )
