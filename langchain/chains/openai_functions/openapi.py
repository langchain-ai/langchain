import json
import re
from collections import defaultdict
from typing import Any, Optional, List, Tuple, Dict, Callable, Union

import requests
from openapi_schema_pydantic import Parameter

from langchain import LLMChain
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
        properties[p.name] = schema.dict(exclude_none=True)
        if p.required:
            required.append(p.name)
    return {"type": "object", "properties": properties, "required": required}


# TODO: Re-implement using APIOperation
def openapi_spec_to_openai_fn(
    spec: OpenAPISpec, *, prefer_short: bool = False
) -> Tuple[List[Dict[str, Any]], Callable]:
    """Convert a valid OpenAPI spec to the JSON Schema format expected for OpenAI
        functions.

    Args:
        spec: The OpenAPI spec to convert
        prefer_short: Whether to use 'summary' or 'description' for a schema when
            both are present.

    Returns:
        Tuple of the OpenAI functions JSON schema and a default function for executing
            a request based on the OpenAI function schema.
    """
    if not spec.paths:
        return [], lambda: None
    functions = []
    _name_to_call_map = {}
    api_name = spec.info.title.replace(" ", "_").replace("-", "_").lower()
    for path in spec.paths:
        path_params = {
            (p.name, p.param_in): p for p in spec.get_parameters_for_path(path)
        }
        for method in spec.get_methods_for_path(path):
            op = spec.get_operation(path, method)
            op_params = path_params.copy()
            for param in spec.get_parameters_for_operation(op):
                op_params[(param.name, param.param_in)] = param
            params_by_type = defaultdict(list)
            for name_loc, p in op_params.items():
                params_by_type[name_loc[1]].append(p)

            url = spec.base_url.rstrip("/") + "/" + path.lstrip("/")
            request_args: Dict = {
                "method": {"const": method.upper()},
                "url": {"const": url},
            }
            param_type_to_arg_name = {
                "query": "params",
                "header": "headers",
                "cookie": "cookies",
                "path": "path_params",
            }
            for param_type, arg_name in param_type_to_arg_name.items():
                if params_by_type[param_type]:
                    request_args[arg_name] = _openapi_params_to_json_schema(
                        params_by_type[param_type], spec
                    )

            request_body = spec.get_request_body_for_operation(op)
            # TODO: Support more MIME types.
            if request_body and request_body.content:
                media_types = []
                for media_type in request_body.content.values():
                    if media_type.media_type_schema:
                        schema = spec.get_schema(media_type.media_type_schema)
                        media_types.append(schema.dict(exclude_none=True))
                if len(media_types) == 1:
                    request_args["data"] = media_types[0]
                elif len(media_types) > 1:
                    request_args["data"] = {"anyOf": media_types}

            if op.operationId:
                name = op.operationId
            else:
                name = api_name + "_" + path.replace("/", "_") + "_" + method
            fn = {
                "name": name,
                "description": _get_description(op, prefer_short),
                "parameters": {
                    "type": "object",
                    "properties": request_args,
                    "required": ["method", "url"],
                },
            }
            functions.append(fn)
            _name_to_call_map[fn["name"]] = {"method": method, "url": url}

    def default_call_api(function_call: dict) -> Any:
        name = function_call["name"]
        args = function_call["arguments"]
        _request_args = args if isinstance(args, dict) else json.loads(args.strip())
        method = _name_to_call_map[name]["method"]
        url = _name_to_call_map[name]["url"]
        path_params = _request_args.pop("path_params", {})
        _format_url(url, path_params)
        if "data" in _request_args and isinstance(_request_args["data"], dict):
            _request_args["data"] = json.dumps(_request_args["data"])
        return requests.request(method, url, **_request_args)

    return functions, default_call_api


def openapi_spec_to_openai_fn2(spec: OpenAPISpec) -> Tuple[List[Dict[str, Any]], Callable]:
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
                media_types = []
                for media_type in request_body.content.values():
                    if media_type.media_type_schema:
                        schema = spec.get_schema(media_type.media_type_schema)
                        media_types.append(schema.dict(exclude_none=True))
                if len(media_types) == 1:
                    request_args["data"] = media_types[0]
                elif len(media_types) > 1:
                    request_args["data"] = {"anyOf": media_types}

            api_op = APIOperation.from_openapi_spec(spec, path, method)
            fn = {
                "name": api_op.operation_id,
                "description": api_op.description,
                "parameters": {
                    "type": "object",
                    "properties": request_args,
                }
            }
            functions.append(fn)
            _name_to_call_map[fn["name"]] = {"method": method, "url": api_op.base_url + api_op.path}

    def default_call_api(function_call: dict, **kwargs: Any) -> Any:
        name = function_call["name"]
        args = function_call["arguments"]
        _request_args = args if isinstance(args, dict) else json.loads(args.strip())
        method = _name_to_call_map[name]["method"]
        url = _name_to_call_map[name]["url"]
        path_params = _request_args.pop("path_params", {})
        _format_url(url, path_params)
        if "data" in _request_args and isinstance(_request_args["data"], dict):
            _request_args["data"] = json.dumps(_request_args["data"])
        _kwargs = {**_request_args, **kwargs}
        return requests.request(method, url, **_kwargs)

    return functions, default_call_api


def get_openapi_chain(spec: Union[OpenAPISpec, str]) -> LLMChain:
    if isinstance(spec, str):
        for conversion in (OpenAPISpec.from_url, OpenAPISpec.from_file, OpenAPISpec.from_text):
            try:
                spec = conversion(spec)
                break
            except:
                pass
