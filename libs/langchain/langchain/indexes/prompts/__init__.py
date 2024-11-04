"""Relevant prompts for constructing indexes."""

from langchain_core._api import warn_deprecated

warn_deprecated(
    since="0.1.47",
    message=(
        "langchain.indexes.prompts will be removed in the future."
        "If you're relying on these prompts, please open an issue on "
        "GitHub to explain your use case."
    ),
    pending=True,
)
