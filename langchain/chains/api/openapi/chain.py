"""Chain that makes API calls and summarizes the responses to answer a question."""
# %%
from __future__ import annotations
import json

from typing import Dict, List, NamedTuple, Optional, Union
from openapi_schema_pydantic import OpenAPI, Operation

from pydantic import BaseModel, Field
from requests import Response
from langchain.chains.api.openapi.requests_chain import APIRequesterChain
from langchain.chains.api.openapi.response_chain import APIResponderChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.requests import Requests


class _ParamMapping(NamedTuple):
    """Mapping from parameter name to parameter value."""

    query_params: List[str]
    body_params: List[str]
    path_params: List[str]


class OpenAPIEndpointChain(Chain, BaseModel):
    """Chain interacts with an OpenAPI endpoint using natural language."""

    api_request_chain: LLMChain
    api_response_chain: LLMChain
    error_handling_chain: Optional[LLMChain] = Field(
        default=None,
    )
    # description: str
    url: str = Field(alias="url")
    method: HTTPVerb = Field(alias="method")
    requests: Requests = Field(exclude=True)
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
        path = self.url
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
        """Use the serialized typescript dictionary to resolve the path, query params dict, and optional requestBody dict."""
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
        api_arguments = self.api_request_chain.predict_and_parse(
            instructions=instructions
        )
        if api_arguments.startswith("ERROR"):
            return {self.output_key: api_arguments}
        elif api_arguments.startswith("MESSAGE:"):
            return {self.output_key: api_arguments[len("MESSAGE:") :]}
        try:
            request_args = self.deserialize_json_input(api_arguments)
            method = getattr(self.requests, self.method.value)
            api_response: Response = method(**request_args)
            if api_response.status_code != 200:
                response_text = (
                    f"{api_response.status_code}: {api_response.reason}"
                    + f"\nFor {self.method.value.upper()}  {request_args['url']}\nCalled with args: {request_args['params']}"
                )
            else:
                response_text = api_response.text
        except Exception as e:
            response_text = f"Error with message {str(e)}"
        answer = self.api_response_chain.predict_and_parse(
            response=response_text,
            instructions=instructions,
        )
        self.callback_manager.on_text(
            answer, color="yellow", end="\n", verbose=self.verbose
        )
        return {self.output_key: answer}

    @property
    def _chain_type(self) -> str:
        return "openapi_chain"

    @classmethod
    def from_operation_and_url(
        cls,
        path: str,
        method: str,
        spec_url: OpenAPI,
        requests: Requests,
        llm: BaseLLM,
        # TODO: Handle async
    ) -> "OpenAPIEndpointChain":
        """Create an OpenAPIEndpoint from a spec at the specified url."""
        spec = get_openapi_spec(spec_url)
        return cls.from_operation_and_spec(
            path=path,
            method=method,
            spec=spec,
            requests=requests,
            llm=llm,
        )

    @classmethod
    def from_operation_and_spec(
        cls,
        path: str,
        method: str,
        spec: OpenAPI,
        requests: Requests,
        llm: BaseLLM,
        # TODO: Handle async
    ) -> "OpenAPIEndpointChain":
        """Create an OpenAPIEndpointChain from an operation and a spec."""
        param_mapping = _ParamMapping(
            query_params=query_params,
            body_params=body_params,
            path_params=path_params,
        )
        # encoding_type stuff
        # response_schema = generate_resolved_response_schema(operation, spec)
        if path_params:
            if not operation_schema.required:
                operation_schema.required = []
            operation_schema.required.extend(path_params)
        requests_chain = APIRequesterChain.from_operation_schema(
            llm,
            operation_schema=operation_schema,
            full_spec=spec,
        )
        response_chain = APIResponderChain.from_llm(llm)
        # operation_id = get_cleaned_operation_id(operation, path, requests_method)
        # description = operation.description or f"Call {operation_id}"
        return cls(
            api_request_chain=requests_chain,
            api_response_chain=response_chain,
            requests=requests,
            url=f"{base_url}{path}",
            method=http_verb,
            param_mapping=param_mapping,
        )


# %%
# CACHED_OPENAPI_SPECS = [
#     "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/spotify.com/1.0.0/openapi.yaml",
#     "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/xkcd.com/1.0.0/openapi.yaml",
#     "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/notion.com/1.0.0/openapi.yaml",
#     "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/twitter.com/current/2.61/openapi.yaml",
# ]
from langchain.llms import OpenAI, Anthropic

# llm = OpenAI()
llm = Anthropic()
requests = Requests()
method = "post"
twitter_url = "http://127.0.0.1:7289/openapi.json"
path = "/hide"
chain = OpenAPIEndpointChain.from_operation_and_url(
    path=path,
    method=method,
    spec_url=twitter_url,
    requests=requests,
    llm=llm,
)

# %%
print(chain("Please hide."))

# %%
chain.url
# %%
