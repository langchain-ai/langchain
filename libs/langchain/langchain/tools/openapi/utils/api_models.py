import sys

from langchain import _PYDANTIC_MAJOR_VERSION

if _PYDANTIC_MAJOR_VERSION == 1:
    from langchain.tools.openapi.utils._api_models_p1 import (
        APIOperation,
        APIRequestBody,
        APIRequestBodyProperty,
    )
else:
    from langchain.tools.openapi.utils._api_models_p2 import (
        APIOperation,
        APIRequestBody,
        APIRequestBodyProperty,
    )

__all__ = ["APIOperation", "APIRequestBody", "APIRequestBodyProperty"]
