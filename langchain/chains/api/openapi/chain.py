"""Chain that makes API calls and summarizes the responses to answer a question."""
from __future__ import annotations

import json
from typing import Dict, List, NamedTuple, Optional, cast

from pydantic import BaseModel, Field
from requests import Response

from langchain.chains.api.openapi.requests_chain import APIRequesterChain
from langchain.chains.api.openapi.response_chain import APIResponderChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.requests import Requests
from langchain.tools.openapi.utils.api_models import APIOperation


class _ParamMapping(NamedTuple):
    """Mapping from parameter name to parameter value."""

    query_params: List[str]
    body_params: List[str]
    path_params: List[str]


class OpenAPIEndpointChain(Chain, BaseModel):
    """Chain interacts with an OpenAPI endpoint using natural language."""

    api_request_chain: LLMChain
    api_response_chain: LLMChain
    api_operation: APIOperation
    requests: Requests = Field(exclude=True, default_factory=Requests)
    param_mapping: _ParamMapping = Field(alias="param_mapping")
    instructions_key: str = "instructions"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.instructions_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _construct_path(self, args: Dict[str, str]) -> str:
        """Construct the path from the deserialized input."""
        path = self.api_operation.base_url + self.api_operation.path
        for param in self.param_mapping.path_params:
            path = path.replace(f"{{{param}}}", args.pop(param, ""))
        return path

    def _extract_query_params(self, args: Dict[str, str]) -> Dict[str, str]:
        """Extract the query params from the deserialized input."""
        query_params = {}
        for param in self.param_mapping.query_params:
            if param in args:
                query_params[param] = args.pop(param)
        return query_params

    def _extract_body_params(self, args: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Extract the request body params from the deserialized input."""
        body_params = None
        if self.param_mapping.body_params:
            body_params = {}
            for param in self.param_mapping.body_params:
                if param in args:
                    body_params[param] = args.pop(param)
        return body_params

    def deserialize_json_input(self, serialized_args: str) -> dict:
        """Use the serialized typescript dictionary.

        Resolve the path, query params dict, and optional requestBody dict.
        """
        args: dict = json.loads(serialized_args)
        path = self._construct_path(args)
        body_params = self._extract_body_params(args)
        query_params = self._extract_query_params(args)
        return {
            "url": path,
            "data": body_params,
            "params": query_params,
        }

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        instructions = inputs[self.instructions_key]
        _api_arguments = self.api_request_chain.predict_and_parse(
            instructions=instructions
        )
        api_arguments = cast(str, _api_arguments)
        if api_arguments.startswith("ERROR"):
            return {self.output_key: api_arguments}
        elif api_arguments.startswith("MESSAGE:"):
            return {self.output_key: api_arguments[len("MESSAGE:") :]}
        try:
            request_args = self.deserialize_json_input(api_arguments)
            method = getattr(self.requests, self.api_operation.method.value)
            api_response: Response = method(**request_args)
            if api_response.status_code != 200:
                method_str = str(self.api_operation.method.value)
                response_text = (
                    f"{api_response.status_code}: {api_response.reason}"
                    + f"\nFor {method_str.upper()}  {request_args['url']}\n"
                    + f"Called with args: {request_args['params']}"
                )
            else:
                response_text = api_response.text
        except Exception as e:
            response_text = f"Error with message {str(e)}"
        _answer = self.api_response_chain.predict_and_parse(
            response=response_text,
            instructions=instructions,
        )
        answer = cast(str, _answer)
        self.callback_manager.on_text(
            answer, color="yellow", end="\n", verbose=self.verbose
        )
        return {self.output_key: answer}

    @classmethod
    def from_url_and_method(
        cls,
        spec_url: str,
        path: str,
        method: str,
        llm: BaseLLM,
        requests: Optional[Requests] = None,
        # TODO: Handle async
    ) -> "OpenAPIEndpointChain":
        """Create an OpenAPIEndpoint from a spec at the specified url."""
        operation = APIOperation.from_openapi_url(spec_url, path, method)
        return cls.from_api_operation(
            operation,
            requests=requests,
            llm=llm,
        )

    @classmethod
    def from_api_operation(
        cls,
        operation: APIOperation,
        llm: BaseLLM,
        requests: Optional[Requests] = None,
        verbose: bool = False
        # TODO: Handle async
    ) -> "OpenAPIEndpointChain":
        """Create an OpenAPIEndpointChain from an operation and a spec."""
        param_mapping = _ParamMapping(
            query_params=operation.query_params,
            body_params=[],  # TODO
            path_params=operation.path_params,
        )
        requests_chain = APIRequesterChain.from_llm_and_typescript(
            llm, typescript_definition=operation.to_typescript(), verbose=verbose
        )
        response_chain = APIResponderChain.from_llm(llm, verbose=verbose)
        _requests = requests or Requests()
        return cls(
            api_request_chain=requests_chain,
            api_response_chain=response_chain,
            api_operation=operation,
            requests=_requests,
            param_mapping=param_mapping,
            verbose=verbose,
        )
