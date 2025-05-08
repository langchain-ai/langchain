from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import requests
from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.utils.input import get_colored_text
from requests import Response

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain

if TYPE_CHECKING:
    from langchain_community.utilities.openapi import OpenAPISpec
    from openapi_pydantic import Parameter


def _get_description(o: Any, prefer_short: bool) -> Optional[str]:
    summary = getattr(o, "summary", None)
    description = getattr(o, "description", None)
    if prefer_short:
        return summary or description
    return description or summary


def _format_url(url: str, path_params: dict) -> str:
    expected_path_param = re.findall(r"{(.*?)}", url)
    new_params = {}
    for param in expected_path_param:
        clean_param = param.lstrip(".;").rstrip("*")
        val = path_params[clean_param]
        if isinstance(val, list):
            if param[0] == ".":
                sep = "." if param[-1] == "*" else ","
                new_val = "." + sep.join(val)
            elif param[0] == ";":
                sep = f"{clean_param}=" if param[-1] == "*" else ","
                new_val = f"{clean_param}=" + sep.join(val)
            else:
                new_val = ",".join(val)
        elif isinstance(val, dict):
            kv_sep = "=" if param[-1] == "*" else ","
            kv_strs = [kv_sep.join((k, v)) for k, v in val.items()]
            if param[0] == ".":
                sep = "."
                new_val = "."
            elif param[0] == ";":
                sep = ";"
                new_val = ";"
            else:
                sep = ","
                new_val = ""
            new_val += sep.join(kv_strs)
        else:
            if param[0] == ".":
                new_val = f".{val}"
            elif param[0] == ";":
                new_val = f";{clean_param}={val}"
            else:
                new_val = val
        new_params[param] = new_val
    return url.format(**new_params)


def _openapi_params_to_json_schema(params: list[Parameter], spec: OpenAPISpec) -> dict:
    properties = {}
    required = []
    for p in params:
        if p.param_schema:
            schema = spec.get_schema(p.param_schema)
        else:
            media_type_schema = list(p.content.values())[0].media_type_schema
            schema = spec.get_schema(media_type_schema)
        if p.description and not schema.description:
            schema.description = p.description
        properties[p.name] = json.loads(schema.json(exclude_none=True))
        if p.required:
            required.append(p.name)
    return {"type": "object", "properties": properties, "required": required}


def openapi_spec_to_openai_fn(
    spec: OpenAPISpec,
) -> tuple[list[dict[str, Any]], Callable]:
    """Convert a valid OpenAPI spec to the JSON Schema format expected for OpenAI
        functions.

    Args:
        spec: OpenAPI spec to convert.

    Returns:
        Tuple of the OpenAI functions JSON schema and a default function for executing
            a request based on the OpenAI function schema.
    """
    try:
        from langchain_community.tools import APIOperation
    except ImportError:
        raise ImportError(
            "Could not import langchain_community.tools. "
            "Please install it with `pip install langchain-community`."
        )

    if not spec.paths:
        return [], lambda: None
    functions = []
    _name_to_call_map = {}
    for path in spec.paths:
        path_params = {
            (p.name, p.param_in): p for p in spec.get_parameters_for_path(path)
        }
        for method in spec.get_methods_for_path(path):
            request_args = {}
            op = spec.get_operation(path, method)
            op_params = path_params.copy()
            for param in spec.get_parameters_for_operation(op):
                op_params[(param.name, param.param_in)] = param
            params_by_type = defaultdict(list)
            for name_loc, p in op_params.items():
                params_by_type[name_loc[1]].append(p)
            param_loc_to_arg_name = {
                "query": "params",
                "header": "headers",
                "cookie": "cookies",
                "path": "path_params",
            }
            for param_loc, arg_name in param_loc_to_arg_name.items():
                if params_by_type[param_loc]:
                    request_args[arg_name] = _openapi_params_to_json_schema(
                        params_by_type[param_loc], spec
                    )
            request_body = spec.get_request_body_for_operation(op)
            # TODO: Support more MIME types.
            if request_body and request_body.content:
                media_types = {}
                for media_type, media_type_object in request_body.content.items():
                    if media_type_object.media_type_schema:
                        schema = spec.get_schema(media_type_object.media_type_schema)
                        media_types[media_type] = json.loads(
                            schema.json(exclude_none=True)
                        )
                if len(media_types) == 1:
                    media_type, schema_dict = list(media_types.items())[0]
                    key = "json" if media_type == "application/json" else "data"
                    request_args[key] = schema_dict
                elif len(media_types) > 1:
                    request_args["data"] = {"anyOf": list(media_types.values())}

            api_op = APIOperation.from_openapi_spec(spec, path, method)
            fn = {
                "name": api_op.operation_id,
                "description": api_op.description,
                "parameters": {
                    "type": "object",
                    "properties": request_args,
                },
            }
            functions.append(fn)
            _name_to_call_map[fn["name"]] = {
                "method": method,
                "url": api_op.base_url + api_op.path,
            }

    def default_call_api(
        name: str,
        fn_args: dict,
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        method = _name_to_call_map[name]["method"]
        url = _name_to_call_map[name]["url"]
        path_params = fn_args.pop("path_params", {})
        url = _format_url(url, path_params)
        if "data" in fn_args and isinstance(fn_args["data"], dict):
            fn_args["data"] = json.dumps(fn_args["data"])
        _kwargs = {**fn_args, **kwargs}
        if headers is not None:
            if "headers" in _kwargs:
                _kwargs["headers"].update(headers)
            else:
                _kwargs["headers"] = headers
        if params is not None:
            if "params" in _kwargs:
                _kwargs["params"].update(params)
            else:
                _kwargs["params"] = params
        return requests.request(method, url, **_kwargs)

    return functions, default_call_api


class SimpleRequestChain(Chain):
    """Chain for making a simple request to an API endpoint."""

    request_method: Callable
    """Method to use for making the request."""
    output_key: str = "response"
    """Key to use for the output of the request."""
    input_key: str = "function"
    """Key to use for the input of the request."""

    @property
    def input_keys(self) -> list[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        """Run the logic of this chain and return the output."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        name = inputs[self.input_key].pop("name")
        args = inputs[self.input_key].pop("arguments")
        _pretty_name = get_colored_text(name, "green")
        _pretty_args = get_colored_text(json.dumps(args, indent=2), "green")
        _text = f"Calling endpoint {_pretty_name} with arguments:\n" + _pretty_args
        _run_manager.on_text(_text)
        api_response: Response = self.request_method(name, args)
        if api_response.status_code != 200:
            response = (
                f"{api_response.status_code}: {api_response.reason}"
                + f"\nFor {name} "
                + f"Called with args: {args.get('params', '')}"
            )
        else:
            try:
                response = api_response.json()
            except Exception:
                response = api_response.text
        return {self.output_key: response}


@deprecated(
    since="0.2.13",
    message=(
        "This function is deprecated and will be removed in langchain 1.0. "
        "See API reference for replacement: "
        "https://api.python.langchain.com/en/latest/chains/langchain.chains.openai_functions.openapi.get_openapi_chain.html"  # noqa: E501
    ),
    removal="1.0",
)
def get_openapi_chain(
    spec: Union[OpenAPISpec, str],
    llm: Optional[BaseLanguageModel] = None,
    prompt: Optional[BasePromptTemplate] = None,
    request_chain: Optional[Chain] = None,
    llm_chain_kwargs: Optional[dict] = None,
    verbose: bool = False,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    **kwargs: Any,
) -> SequentialChain:
    """Create a chain for querying an API from a OpenAPI spec.

    Note: this class is deprecated. See below for a replacement implementation.
        The benefits of this implementation are:

        - Uses LLM tool calling features to encourage properly-formatted API requests;
        - Includes async support.

        .. code-block:: python

            from typing import Any

            from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
            from langchain_community.utilities.openapi import OpenAPISpec
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI

            # Define API spec. Can be JSON or YAML
            api_spec = \"\"\"
            {
            "openapi": "3.1.0",
            "info": {
                "title": "JSONPlaceholder API",
                "version": "1.0.0"
            },
            "servers": [
                {
                "url": "https://jsonplaceholder.typicode.com"
                }
            ],
            "paths": {
                "/posts": {
                "get": {
                    "summary": "Get posts",
                    "parameters": [
                    {
                        "name": "_limit",
                        "in": "query",
                        "required": false,
                        "schema": {
                        "type": "integer",
                        "example": 2
                        },
                        "description": "Limit the number of results"
                    }
                    ]
                }
                }
            }
            }
            \"\"\"

            parsed_spec = OpenAPISpec.from_text(api_spec)
            openai_fns, call_api_fn = openapi_spec_to_openai_fn(parsed_spec)
            tools = [
                {"type": "function", "function": fn}
                for fn in openai_fns
            ]

            prompt = ChatPromptTemplate.from_template(
                "Use the provided APIs to respond to this user query:\\n\\n{query}"
            )
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

            def _execute_tool(message) -> Any:
                if tool_calls := message.tool_calls:
                    tool_call = message.tool_calls[0]
                    response = call_api_fn(name=tool_call["name"], fn_args=tool_call["args"])
                    response.raise_for_status()
                    return response.json()
                else:
                    return message.content

            chain = prompt | llm | _execute_tool

        .. code-block:: python

            response = chain.invoke({"query": "Get me top two posts."})

    Args:
        spec: OpenAPISpec or url/file/text string corresponding to one.
        llm: language model, should be an OpenAI function-calling model, e.g.
            `ChatOpenAI(model="gpt-3.5-turbo-0613")`.
        prompt: Main prompt template to use.
        request_chain: Chain for taking the functions output and executing the request.
    """  # noqa: E501
    try:
        from langchain_community.utilities.openapi import OpenAPISpec
    except ImportError as e:
        raise ImportError(
            "Could not import langchain_community.utilities.openapi. "
            "Please install it with `pip install langchain-community`."
        ) from e
    if isinstance(spec, str):
        for conversion in (
            OpenAPISpec.from_url,
            OpenAPISpec.from_file,
            OpenAPISpec.from_text,
        ):
            try:
                spec = conversion(spec)
                break
            except ImportError as e:
                raise e
            except Exception:
                pass
        if isinstance(spec, str):
            raise ValueError(f"Unable to parse spec from source {spec}")
    openai_fns, call_api_fn = openapi_spec_to_openai_fn(spec)
    if not llm:
        raise ValueError(
            "Must provide an LLM for this chain.For example,\n"
            "from langchain_openai import ChatOpenAI\n"
            "llm = ChatOpenAI()\n"
        )
    prompt = prompt or ChatPromptTemplate.from_template(
        "Use the provided API's to respond to this user query:\n\n{query}"
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs={"functions": openai_fns},
        output_parser=JsonOutputFunctionsParser(args_only=False),
        output_key="function",
        verbose=verbose,
        **(llm_chain_kwargs or {}),
    )
    request_chain = request_chain or SimpleRequestChain(
        request_method=lambda name, args: call_api_fn(
            name, args, headers=headers, params=params
        ),
        verbose=verbose,
    )
    return SequentialChain(
        chains=[llm_chain, request_chain],
        input_variables=llm_chain.input_keys,
        output_variables=["response"],
        verbose=verbose,
        **kwargs,
    )
