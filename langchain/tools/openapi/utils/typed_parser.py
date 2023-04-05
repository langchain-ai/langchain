"""Utility functions for parsing an OpenAPI spec into LangChain Tools / Toolkits."""
import copy
from enum import Enum
import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union
import requests

from openapi_schema_pydantic import (
    MediaType,
    OpenAPI,
    Operation,
    Parameter,
    Reference,
    RequestBody,
    Response,
    Schema,
)
from openapi_schema_pydantic import OpenAPI
from pydantic import BaseModel, ValidationError, constr
import yaml

from langchain.requests import RequestsWrapper

logger = logging.getLogger(__name__)


class OpenAPISpec(OpenAPI):
    """OpenAPI Model that removes misformatted parts of the spec."""

    @classmethod
    def parse_obj(cls, obj):
        try:
            return super().parse_obj(obj)
        except ValidationError as e:
            # We are handling possibly misconfigured specs and want to do a best-effort
            # job to get a reasonable interface out of it.
            new_obj = copy.deepcopy(obj)
            for error in e.errors():
                keys = error["loc"]
                item = new_obj
                for key in keys[:-1]:
                    item = item[key]
                item.pop(keys[-1], None)
            return cls.parse_obj(new_obj)

    @classmethod
    def from_spec_dict(cls, spec_dict: dict) -> "OpenAPISpec":
        """Get an OpenAPI spec from a dict."""
        return cls.parse_obj(spec_dict)

    @classmethod
    def from_url(cls, url: str) -> "OpenAPISpec":
        """Get an OpenAPI spec from a URL."""
        response = requests.get(url)
        try:
            open_api_spec = json.loads(response.text)
        except json.JSONDecodeError:
            open_api_spec = yaml.safe_load(response.text)
        return cls.from_spec_dict(open_api_spec)


class _Joiner(str, Enum):
    union = "|"
    intersection = "&"


class JoinedType(BaseModel):

    sub_types: List[str]

    joiner: _Joiner


## Resolving


def _resolve_reference(ref: Reference, spec: OpenAPI, seen: Set[str]) -> str:
    """Resolve a $ref to a definition in the OpenAPI spec."""
    ref_name = ref.split("/")[-1]
    # TODO: The parsing library loses the `required` tags in the spec.
    if ref_name in spec.components.schemas:
        schema = spec.components.schemas[ref_name]
        if ref_name not in seen:
            seen.add(ref_name)
            _dereference_children(schema, spec, seen)
        return ref_name
    # TODO: These probably also need recursive dereferencing
    component_types = [
        spec.components.parameters,
        spec.components.requestBodies,
        # spec.components.responses,
    ]
    for component_type in component_types:
        if component_type is not None and ref_name in component_type:
            return component_type[ref_name]
    raise ValueError(f"Reference {ref} not found in spec")


def _dereference_anyof(
    schema: Schema, spec: OpenAPI, seen: Set[str]
) -> Optional[JoinedType]:
    """Dereference anyOf schemas."""
    if not schema.anyOf:
        return
    resolved_any_of = []
    for any_of_schema in schema.anyOf:
        if isinstance(any_of_schema, Reference):
            resolved_any_of.append(_resolve_reference(any_of_schema.ref, spec, seen))
        elif isinstance(schema.type, str):
            resolved_any_of.append(any_of_schema.type)
        else:
            resolved_any_of.append(any_of_schema)
    seen.add(JoinedType(sub_types=resolved_any_of, joiner=_Joiner.intersection))


def _dereference_properties(schema: Schema, spec: OpenAPI, seen: Set[str]) -> None:
    """Dereference properties."""
    if schema.properties:
        resolved_properties = {}
        for prop_name, prop_schema in schema.properties.items():
            if isinstance(prop_schema, Reference):
                resolved_properties[prop_name] = _resolve_reference(
                    prop_schema.ref, spec, seen
                )
            else:
                resolved_properties[prop_name] = prop_schema
            _dereference_children(resolved_properties[prop_name], spec, seen)


def _dereference_children(schema: Schema, spec: OpenAPI, seen: Set[str]) -> None:
    """Dereference children."""
    _dereference_anyof(schema, spec, seen)
    _dereference_properties(schema, spec, seen)


###### Shared functions #######


def extract_path_params(path: str) -> List[str]:
    """Extract path parameters from a URI path like /path/to/{user_id}."""
    path_params_pattern = r"{(.*?)}"
    return re.findall(path_params_pattern, path)


def extract_query_params(operation: Operation) -> List[str]:
    """Extract parameter names from the request query of an operation."""
    query_params = []
    if operation.parameters is not None:
        for param in operation.parameters:
            if isinstance(param, Reference):
                name = param.ref.split("/")[-1]
            else:
                name = param.name
            query_params.append(name)

    return query_params


def extract_body_params(operation: Operation, spec: OpenAPI) -> List[str]:
    """Extract parameter names from the request body of an operation."""
    body_params = []
    request_body = operation.requestBody
    if request_body is None:
        return body_params

    if isinstance(request_body, Reference):
        name = request_body.ref.split("/")[-1]
        body_params.append(name)
        return body_params
    for content_type, json_content in request_body.content.items():
        media_type_schema = json_content.media_type_schema
        if media_type_schema is None:
            logger.debug(
                f"Content type '{content_type}' not supported"
                f" for operation {operation.operationId}."
                f"Supported types: {request_body.content.keys()}"
            )
            continue
        media_type_schema = resolve_schema(media_type_schema, spec)
        if media_type_schema.anyOf:
            for _schema_anyof in media_type_schema.anyOf:
                body_params.extend(_schema_anyof.properties.keys())
        elif media_type_schema.properties:
            body_params.extend(media_type_schema.properties.keys())
        else:
            logger.warning(
                f"No properties found for {media_type_schema}."
                " oneOf, allOf, and other attributes not yet implemented."
            )
    return body_params


def extract_query_and_body_params(
    operation: Operation, spec: OpenAPI
) -> Tuple[List[str], List[str]]:
    """Extract query and body parameters from an operation."""
    query_params = extract_query_params(operation)
    body_params = extract_body_params(operation, spec)
    return query_params, body_params


def generate_resolved_schema(
    operation: Operation, spec: OpenAPI
) -> Tuple[Schema, Optional[str]]:
    """Generate a combined schema object, dereferencing any references."""
    request_body_schema, encoding_type = _resolve_request_body_schema(operation, spec)
    query_params_schema = _resolve_query_params_schema(operation, spec)

    combined_schema = query_params_schema
    if request_body_schema:
        if request_body_schema.anyOf:
            for schema in request_body_schema.anyOf:
                combined_schema.properties.update(schema.properties)
        elif request_body_schema.properties:
            combined_schema.properties.update(request_body_schema.properties)
        else:
            logger.warning(
                "No properties in request body schema for operation"
                f" {operation.operationId}\n"
                f" oneOf, allOf, and other attributes not yet implemented."
            )
    return combined_schema, encoding_type


def _resolve_response(response: Response, spec: OpenAPI) -> Dict[str, Schema]:
    """Resolve a response object."""
    if response.content is None:
        return response
    return _resolve_media_type_schema(response.content, spec)


def generate_resolved_response_schema(
    operation: Operation, spec: OpenAPI
) -> Optional[MediaType]:
    """Generate a combined schema object, dereferencing any references."""
    if not operation.responses:
        return None
    response_schema = operation.responses.get("200")
    if response_schema is None:
        return None
    if isinstance(response_schema, Reference):
        # TODO: Not the right type actually
        response_schema = _resolve_reference(response_schema.ref, spec)
    if not isinstance(response_schema, Response):
        return
    schema_dict = _resolve_response(response_schema, spec)
    if not response_schema:
        return
    if "application/json" not in schema_dict:
        return
    json_schema = schema_dict["application/json"]
    resolved_schema = resolve_schema(json_schema, spec)
    return resolved_schema


def get_cleaned_operation_id(operation: Operation, path: str, verb: str) -> str:
    """Get a cleaned operation id from an operation id."""
    operation_id = operation.operationId
    if operation_id is None:
        # Replace all punctuation of any kind with underscore
        path = re.sub(r"[^a-zA-Z0-9]", "_", path.lstrip("/"))
        operation_id = f"{path}{verb.upper()}"
    return operation_id.replace("-", "_").replace(".", "_").replace("/", "_")


def resolve_schema(
    schema: Union[Schema, Reference],
    spec: OpenAPI,
    description: Optional[str] = None,
) -> Schema:
    """Resolve a schema or ref to a definition in the OpenAPI spec if needed."""
    _schema = schema
    if isinstance(schema, Reference):
        # TODO: the typing here is off since
        # the result of _resolve_reference may be a
        # parameter, reference, or response
        _schema = _resolve_reference(schema.ref, spec)
    if description and not _schema.description:
        _schema.description = description
    _dereference_children(_schema, spec)
    return _schema


# Query params and Path params can't be nested deeply!
def get_(path: str, method: str, spec: OpenAPI) -> None:
    """Foo."""
    path_item = spec.paths.get(path)
    if not path_item:
        return
    operation = getattr(path_item, method, None)
    if not isinstance(operation, Operation):
        return
    query_params = []
    path_params = []
    data_models = set()
    if operation.parameters:
        for parameter in operation.parameters:
            if isinstance(parameter, Reference):
                parameter = _resolve_reference(parameter.ref, spec)
            if parameter.param_in == "path":
                path_params.append(parameter)
                pass
            elif parameter.param_in == "query":
                query_params.append(parameter)
                pass
            elif parameter.param_in == "header":
                logger.warning("Header parameters not yet implemented")
                continue
            else:  # parameter.param_in == "cookie"
                logger.warning("Cookie parameters not yet implemented")
                continue
    path_params = extract_path_params(path)
    query_params, body_params = extract_query_and_body_params(operation, spec)
    # operation_schema, encoding_type = generate_resolved_schema(operation, spec)
    # operation.p


if __name__ == "__main__":
    CACHED_OPENAPI_SPECS = ["http://127.0.0.1:7289/openapi.json"]
    spec = OpenAPISpec.from_url(CACHED_OPENAPI_SPECS[0])
    # op = foo("/goto/{x}/{y}/{z}", "post", spec)
    op = foo("/get_state", "get", spec)
    print(spec)
