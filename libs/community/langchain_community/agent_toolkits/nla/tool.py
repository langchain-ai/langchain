"""Tool for interacting with a single API with natural language definition."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool

from langchain_community.chains.openapi.chain import OpenAPIEndpointChain
from langchain_community.tools.openapi.utils.api_models import APIOperation
from langchain_community.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain_community.utilities.requests import Requests


class NLATool(Tool):  # type: ignore[override]
    """Natural Language API Tool."""

    @classmethod
    def from_open_api_endpoint_chain(
        cls, chain: OpenAPIEndpointChain, api_title: str
    ) -> "NLATool":
        """Convert an endpoint chain to an API endpoint tool.

        Args:
            chain: The endpoint chain.
            api_title: The title of the API.

        Returns:
            The API endpoint tool.
        """
        expanded_name = (
            f"{api_title.replace(' ', '_')}.{chain.api_operation.operation_id}"
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
        """Instantiate the tool from the specified path and method.

        Args:
            llm: The language model to use.
            path: The path of the API.
            method: The method of the API.
            spec: The OpenAPI spec.
            requests: Optional requests object. Default is None.
            verbose: Whether to print verbose output. Default is False.
            return_intermediate_steps: Whether to return intermediate steps.
                Default is False.
            kwargs: Additional arguments.

        Returns:
            The tool.
        """
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
