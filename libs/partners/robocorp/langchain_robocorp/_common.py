from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.utils.json_schema import dereference_refs


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
    endpoints: List[Tuple[str, str, dict]]


def reduce_openapi_spec(url: str, spec: dict) -> ReducedOpenAPISpec:
    """Simplify OpenAPI spec to only required information for the agent"""

    # 1. Consider only GET and POST
    endpoints = [
        (route, docs.get("description"), docs)
        for route, operation in spec["paths"].items()
        for operation_name, docs in operation.items()
        if operation_name in ["get", "post"]
    ]

    # 2. Replace any refs so that complete docs are retrieved.
    # Note: probably want to do this post-retrieval, it blows up the size of the spec.
    endpoints = [
        (name, description, dereference_refs(docs, full_schema=spec))
        for name, description, docs in endpoints
    ]

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
        (name, description, reduce_endpoint_docs(docs))
        for name, description, docs in endpoints
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


def get_required_param_descriptions(endpoint_spec: dict) -> str:
    """Get an OpenAPI endpoint required parameter descriptions"""
    descriptions = []

    schema = (
        endpoint_spec.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )
    properties = schema.get("properties", {})

    required_fields = schema.get("required", [])

    for key, value in properties.items():
        if "description" in value:
            if value.get("required") or key in required_fields:
                descriptions.append(value.get("description"))

    return ", ".join(descriptions)

