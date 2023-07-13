"""Tool for interacting with a single API with natural language efinition."""


from typing import Any, Optional

from langchain.agents.tools import Tool
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain.requests import Requests
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec


class NLATool(Tool):
    """Natural Language API Tool."""

    @classmethod
    def from_open_api_endpoint_chain(
        cls, chain: OpenAPIEndpointChain, api_title: str
    ) -> "NLATool":
        """Convert an endpoint chain to an API endpoint tool."""
        expanded_name = (
            f'{api_title.replace(" ", "_")}.{chain.api_operation.operation_id}'
        )
        description = (
            f"I'm an AI from {api_title}. Instruct what you want,"
            " and I'll assist via an API with description:"
            f" {chain.api_operation.description}"
        )
        return cls(name=expanded_name, func=chain.run, description=description)

    @classmethod
    def from_llm_and_method(
        cls,
        llm: BaseLanguageModel,
        path: str,
        method: str,
        spec: OpenAPISpec,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        **kwargs: Any,
    ) -> "NLATool":
        """Instantiate the tool from the specified path and method."""
        api_operation = APIOperation.from_openapi_spec(spec, path, method)
        chain = OpenAPIEndpointChain.from_api_operation(
            api_operation,
            llm,
            requests=requests,
            verbose=verbose,
            return_intermediate_steps=return_intermediate_steps,
            **kwargs,
        )
        return cls.from_open_api_endpoint_chain(chain, spec.info.title)
