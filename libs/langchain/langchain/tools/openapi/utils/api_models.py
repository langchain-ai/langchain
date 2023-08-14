import sys

from langchain import _PYDANTIC_MAJOR_VERSION

if _PYDANTIC_MAJOR_VERSION == 1:
    import langchain.tools.openapi.utils._api_models_p1 as module_to_use
else:
    import langchain.tools.openapi.utils._api_models_p2 as module_to_use


thismodule = sys.modules[__name__]

members = dir(module_to_use)

for member in members:
    setattr(thismodule, member, getattr(module_to_use, member))
