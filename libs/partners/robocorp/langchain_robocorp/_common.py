import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union, cast

from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    create_model,
)
from langchain_core.utils.json_schema import dereference_refs
from langchain_core.utils.pydantic import is_basemodel_instance


@dataclass(frozen=True)
class ReducedOpenAPISpec:
    """A reduced OpenAPI spec.

    This is reduced representation for OpenAPI specs.

    Attributes:
        servers: The servers in the spec.
        description: The description of the spec.
        endpoints: The endpoints in the spec.
    """

    servers: List[dict]
    description: str
    endpoints: List[Tuple[str, dict]]


def reduce_openapi_spec(url: str, spec: dict) -> ReducedOpenAPISpec:
    """Simplify OpenAPI spec to only required information for the agent"""

    # 1. Consider only GET and POST
    endpoints = [
        (route, docs)
        for route, operation in spec["paths"].items()
        for operation_name, docs in operation.items()
        if operation_name in ["get", "post"]
    ]

    # 2. Replace any refs so that complete docs are retrieved.
    # Note: probably want to do this post-retrieval, it blows up the size of the spec.

    # 3. Strip docs down to required request args + happy path response.
    def reduce_endpoint_docs(docs: dict) -> dict:
        out = {}
        if docs.get("summary"):
            out["summary"] = docs.get("summary")
        if docs.get("operationId"):
            out["operationId"] = docs.get("operationId")
        if docs.get("description"):
            out["description"] = docs.get("description")
        if docs.get("parameters"):
            out["parameters"] = [
                parameter
                for parameter in docs.get("parameters", [])
                if parameter.get("required")
            ]
        if "200" in docs["responses"]:
            out["responses"] = docs["responses"]["200"]
        if docs.get("requestBody"):
            out["requestBody"] = docs.get("requestBody")
        return out

    endpoints = [
        (name, reduce_endpoint_docs(dereference_refs(docs, full_schema=spec)))
        for name, docs in endpoints
    ]

    return ReducedOpenAPISpec(
        servers=[
            {
                "url": url,
            }
        ],
        description=spec["info"].get("description", ""),
        endpoints=endpoints,
    )


type_mapping = {
    "string": str,
    "integer": int,
    "number": float,
    "object": dict,
    "array": list,
    "boolean": bool,
    "null": type(None),
}


def get_schema(endpoint_spec: dict) -> dict:
    return (
        endpoint_spec.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )


def create_field(
    schema: dict, required: bool, created_model_names: Set[str]
) -> Tuple[Any, Any]:
    """
    Creates a Pydantic field based on the schema definition.
    """
    if "anyOf" in schema:
        field_types = [
            create_field(sub_schema, required, created_model_names)[0]
            for sub_schema in schema["anyOf"]
        ]
        if len(field_types) == 1:
            field_type = field_types[0]  # Simplified handling
        else:
            field_type = Union[tuple(field_types)]
    else:
        field_type = type_mapping.get(schema.get("type", "string"), str)

    description = schema.get("description", "")

    # Handle nested objects
    if schema.get("type") == "object":
        nested_fields = {
            k: create_field(v, k in schema.get("required", []), created_model_names)
            for k, v in schema.get("properties", {}).items()
        }
        model_name = schema.get("title", f"NestedModel{time.time()}")
        if model_name in created_model_names:
            # needs to be unique
            model_name = model_name + str(time.time())
        nested_model = create_model(model_name, **nested_fields)  # type: ignore
        created_model_names.add(model_name)
        return nested_model, Field(... if required else None, description=description)

    # Handle arrays
    elif schema.get("type") == "array":
        item_type, _ = create_field(
            schema["items"], required=True, created_model_names=created_model_names
        )
        return List[item_type], Field(  # type: ignore
            ... if required else None, description=description
        )

    # Other types
    return field_type, Field(... if required else None, description=description)


def get_param_fields(endpoint_spec: dict) -> dict:
    """Get an OpenAPI endpoint parameter details"""
    schema = get_schema(endpoint_spec)
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    fields = {}
    created_model_names: Set[str] = set()
    for key, value in properties.items():
        is_required = key in required_fields
        field_info = create_field(value, is_required, created_model_names)
        fields[key] = field_info

    return fields


def model_to_dict(
    item: Union[BaseModel, List, Dict[str, Any]],
) -> Any:
    if is_basemodel_instance(item):
        return cast(BaseModel, item).dict()
    elif isinstance(item, dict):
        return {key: model_to_dict(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [model_to_dict(element) for element in item]
    else:
        return item
