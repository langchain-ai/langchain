"""Chain that interprets a prompt and executes python code to do math.

Heavily borrowed from llm_math, wrapper for SymPy
"""
from langchain._api import warn_deprecated

warn_deprecated(
    since="0.0.304",
    message=(
        "On 2023-10-06 this module will be moved to langchain-experimental as "
        "it relies on sympy https://github.com/sympy/sympy/issues/10805"
    ),
    pending=True,
)
