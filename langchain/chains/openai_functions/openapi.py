import json
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests
from openapi_schema_pydantic import Parameter
from requests import Response

from langchain import BasePromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.sequential import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.input import get_colored_text
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.tools import APIOperation
from langchain.utilities.openapi import OpenAPISpec


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


def _openapi_params_to_json_schema(params: List[Parameter], spec: OpenAPISpec) -> dict:
    properties = {}
    required = []
    for p in params:
        if p.param_schema:
            schema = spec.get_schema(p.param_schema)
        else:
            media_type_schema = list(p.content.values())[0].media_type_schema  # type: ignore  # noqa: E501
            schema = spec.get_schema(media_type_schema)
        if p.description and not schema.description:
            schema.description = p.description
        properties[p.name] = json.loads(schema.json(exclude_none=True))
        if p.required:
            required.append(p.name)
    return {"type": "object", "properties": properties, "required": required}


def openapi_spec_to_openai_fn(
    spec: OpenAPISpec,
) -> Tuple[List[Dict[str, Any]], Callable]:
    """Convert a valid OpenAPI spec to the JSON Schema format expected for OpenAI
        functions.

    Args:
        spec: OpenAPI spec to convert.

    Returns:
        Tuple of the OpenAI functions JSON schema and a default function for executing
            a request based on the OpenAI function schema.
    """
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
    request_method: Callable
    output_key: str = "response"
    input_key: str = "function"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the logic of this chain and return the output."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        name = inputs["function"].pop("name")
        args = inputs["function"].pop("arguments")
        _pretty_name = get_colored_text(name, "green")
        _pretty_args = get_colored_text(json.dumps(args, indent=2), "green")
        _text = f"Calling endpoint {_pretty_name} with arguments:\n" + _pretty_args
        _run_manager.on_text(_text)
        api_response: Response = self.request_method(name, args)
        if api_response.status_code != 200:
            response = (
                f"{api_response.status_code}: {api_response.reason}"
                + f"\nFor {name} "
                + f"Called with args: {args['params']}"
            )
        else:
            try:
                response = api_response.json()
            except Exception:  # noqa: E722
                response = api_response.text
        return {self.output_key: response}


def get_openapi_chain(
    spec: Union[OpenAPISpec, str],
    llm: Optional[BaseLanguageModel] = None,
    prompt: Optional[BasePromptTemplate] = None,
    request_chain: Optional[Chain] = None,
    llm_kwargs: Optional[Dict] = None,
    verbose: bool = False,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    **kwargs: Any,
) -> SequentialChain:
    """Create a chain for querying an API from a OpenAPI spec.

    Args:
        spec: OpenAPISpec or url/file/text string corresponding to one.
        llm: language model, should be an OpenAI function-calling model, e.g.
            `ChatOpenAI(model="gpt-3.5-turbo-0613")`.
        prompt: Main prompt template to use.
        request_chain: Chain for taking the functions output and executing the request.
    """
    if isinstance(spec, str):
        for conversion in (
            OpenAPISpec.from_url,
            OpenAPISpec.from_file,
            OpenAPISpec.from_text,
        ):
            try:
                spec = conversion(spec)  # type: ignore[arg-type]
                break
            except Exception:  # noqa: E722
                pass
        if isinstance(spec, str):
            raise ValueError(f"Unable to parse spec from source {spec}")
    openai_fns, call_api_fn = openapi_spec_to_openai_fn(spec)
    llm = llm or ChatOpenAI(
        model="gpt-3.5-turbo-0613",
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
        **(llm_kwargs or {}),
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
