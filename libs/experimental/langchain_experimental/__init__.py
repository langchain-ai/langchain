import importlib
import sys

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
    pydantic_v1 = importlib.import_module("pydantic.v1")
except ImportError:
    pydantic_v1 = importlib.import_module("pydantic")

if "pydantic_v1" not in sys.modules:
    # Use a conditional because langchain experimental
    # will use the same strategy to add pydantic_v1 to sys.modules
    # and may run prior to langchain core package.
    sys.modules["pydantic_v1"] = pydantic_v1
