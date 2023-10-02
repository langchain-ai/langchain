"""Chain that interprets a prompt and executes bash code to perform bash operations."""
from langchain._api import warn_deprecated

warn_deprecated(
    since="0.0.306",
    message="On 2023-10-09 the LLMBashChain will be moved to langchain-experimental",
    pending=True,
)
