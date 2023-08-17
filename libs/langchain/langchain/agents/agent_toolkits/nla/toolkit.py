from __future__ import annotations

from typing import Any, List, Optional, Sequence

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.agents.agent_toolkits.nla.tool import NLATool
from langchain.pydantic_v1 import Field
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain.tools.plugin import AIPlugin
from langchain.utilities.requests import Requests


class NLAToolkit(BaseToolkit):
    """Natural Language API Toolkit."""

    nla_tools: Sequence[NLATool] = Field(...)
    """List of API Endpoint Tools."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools for all the API operations."""
        return list(self.nla_tools)

    @staticmethod
    def _get_http_operation_tools(
        llm: BaseLanguageModel,
        spec: OpenAPISpec,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[NLATool]:
        """Get the tools for all the API operations."""
        if not spec.paths:
            return []
        http_operation_tools = []
        for path in spec.paths:
            for method in spec.get_methods_for_path(path):
                endpoint_tool = NLATool.from_llm_and_method(
                    llm=llm,
                    path=path,
                    method=method,
                    spec=spec,
                    requests=requests,
                    verbose=verbose,
                    **kwargs,
                )
                http_operation_tools.append(endpoint_tool)
        return http_operation_tools

    @classmethod
    def from_llm_and_spec(
        cls,
        llm: BaseLanguageModel,
        spec: OpenAPISpec,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> NLAToolkit:
        """Instantiate the toolkit by creating tools for each operation."""
        http_operation_tools = cls._get_http_operation_tools(
            llm=llm, spec=spec, requests=requests, verbose=verbose, **kwargs
        )
        return cls(nla_tools=http_operation_tools)

    @classmethod
    def from_llm_and_url(
        cls,
        llm: BaseLanguageModel,
        open_api_url: str,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> NLAToolkit:
        """Instantiate the toolkit from an OpenAPI Spec URL"""
        spec = OpenAPISpec.from_url(open_api_url)
        return cls.from_llm_and_spec(
            llm=llm, spec=spec, requests=requests, verbose=verbose, **kwargs
        )

    @classmethod
    def from_llm_and_ai_plugin(
        cls,
        llm: BaseLanguageModel,
        ai_plugin: AIPlugin,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> NLAToolkit:
        """Instantiate the toolkit from an OpenAPI Spec URL"""
        spec = OpenAPISpec.from_url(ai_plugin.api.url)
        # TODO: Merge optional Auth information with the `requests` argument
        return cls.from_llm_and_spec(
            llm=llm,
            spec=spec,
            requests=requests,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def from_llm_and_ai_plugin_url(
        cls,
        llm: BaseLanguageModel,
        ai_plugin_url: str,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> NLAToolkit:
        """Instantiate the toolkit from an OpenAPI Spec URL"""
        plugin = AIPlugin.from_url(ai_plugin_url)
        return cls.from_llm_and_ai_plugin(
            llm=llm, ai_plugin=plugin, requests=requests, verbose=verbose, **kwargs
        )
