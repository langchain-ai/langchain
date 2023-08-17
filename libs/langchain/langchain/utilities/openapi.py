"""Utility functions for parsing an OpenAPI spec."""
from __future__ import annotations

import copy
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
import yaml

from langchain.pydantic_v1 import _PYDANTIC_MAJOR_VERSION, ValidationError

logger = logging.getLogger(__name__)


class HTTPVerb(str, Enum):
    """Enumerator of the HTTP verbs."""

    GET = "get"
    PUT = "put"
    POST = "post"
    DELETE = "delete"
    OPTIONS = "options"
    HEAD = "head"
    PATCH = "patch"
    TRACE = "trace"

    @classmethod
    def from_str(cls, verb: str) -> HTTPVerb:
        """Parse an HTTP verb."""
        try:
            return cls(verb)
        except ValueError:
            raise ValueError(f"Invalid HTTP verb. Valid values are {cls.__members__}")


if _PYDANTIC_MAJOR_VERSION == 1:
    from openapi_schema_pydantic import (
        Components,
        OpenAPI,
        Operation,
        Parameter,
        PathItem,
        Paths,
        Reference,
        RequestBody,
        Schema,
    )

    class OpenAPISpec(OpenAPI):
        """OpenAPI Model that removes mis-formatted parts of the spec."""

        @property
        def _paths_strict(self) -> Paths:
            if not self.paths:
                raise ValueError("No paths found in spec")
            return self.paths

        def _get_path_strict(self, path: str) -> PathItem:
            path_item = self._paths_strict.get(path)
            if not path_item:
                raise ValueError(f"No path found for {path}")
            return path_item

        @property
        def _components_strict(self) -> Components:
            """Get components or err."""
            if self.components is None:
                raise ValueError("No components found in spec. ")
            return self.components

        @property
        def _parameters_strict(self) -> Dict[str, Union[Parameter, Reference]]:
            """Get parameters or err."""
            parameters = self._components_strict.parameters
            if parameters is None:
                raise ValueError("No parameters found in spec. ")
            return parameters

        @property
        def _schemas_strict(self) -> Dict[str, Schema]:
            """Get the dictionary of schemas or err."""
            schemas = self._components_strict.schemas
            if schemas is None:
                raise ValueError("No schemas found in spec. ")
            return schemas

        @property
        def _request_bodies_strict(self) -> Dict[str, Union[RequestBody, Reference]]:
            """Get the request body or err."""
            request_bodies = self._components_strict.requestBodies
            if request_bodies is None:
                raise ValueError("No request body found in spec. ")
            return request_bodies

        def _get_referenced_parameter(
            self, ref: Reference
        ) -> Union[Parameter, Reference]:
            """Get a parameter (or nested reference) or err."""
            ref_name = ref.ref.split("/")[-1]
            parameters = self._parameters_strict
            if ref_name not in parameters:
                raise ValueError(f"No parameter found for {ref_name}")
            return parameters[ref_name]

        def _get_root_referenced_parameter(self, ref: Reference) -> Parameter:
            """Get the root reference or err."""
            parameter = self._get_referenced_parameter(ref)
            while isinstance(parameter, Reference):
                parameter = self._get_referenced_parameter(parameter)
            return parameter

        def get_referenced_schema(self, ref: Reference) -> Schema:
            """Get a schema (or nested reference) or err."""
            ref_name = ref.ref.split("/")[-1]
            schemas = self._schemas_strict
            if ref_name not in schemas:
                raise ValueError(f"No schema found for {ref_name}")
            return schemas[ref_name]

        def get_schema(self, schema: Union[Reference, Schema]) -> Schema:
            if isinstance(schema, Reference):
                return self.get_referenced_schema(schema)
            return schema

        def _get_root_referenced_schema(self, ref: Reference) -> Schema:
            """Get the root reference or err."""
            schema = self.get_referenced_schema(ref)
            while isinstance(schema, Reference):
                schema = self.get_referenced_schema(schema)
            return schema

        def _get_referenced_request_body(
            self, ref: Reference
        ) -> Optional[Union[Reference, RequestBody]]:
            """Get a request body (or nested reference) or err."""
            ref_name = ref.ref.split("/")[-1]
            request_bodies = self._request_bodies_strict
            if ref_name not in request_bodies:
                raise ValueError(f"No request body found for {ref_name}")
            return request_bodies[ref_name]

        def _get_root_referenced_request_body(
            self, ref: Reference
        ) -> Optional[RequestBody]:
            """Get the root request Body or err."""
            request_body = self._get_referenced_request_body(ref)
            while isinstance(request_body, Reference):
                request_body = self._get_referenced_request_body(request_body)
            return request_body

        @staticmethod
        def _alert_unsupported_spec(obj: dict) -> None:
            """Alert if the spec is not supported."""
            warning_message = (
                " This may result in degraded performance."
                + " Convert your OpenAPI spec to 3.1.* spec"
                + " for better support."
            )
            swagger_version = obj.get("swagger")
            openapi_version = obj.get("openapi")
            if isinstance(openapi_version, str):
                if openapi_version != "3.1.0":
                    logger.warning(
                        f"Attempting to load an OpenAPI {openapi_version}"
                        f" spec. {warning_message}"
                    )
                else:
                    pass
            elif isinstance(swagger_version, str):
                logger.warning(
                    f"Attempting to load a Swagger {swagger_version}"
                    f" spec. {warning_message}"
                )
            else:
                raise ValueError(
                    "Attempting to load an unsupported spec:"
                    f"\n\n{obj}\n{warning_message}"
                )

        @classmethod
        def parse_obj(cls, obj: dict) -> OpenAPISpec:
            try:
                cls._alert_unsupported_spec(obj)
                return super().parse_obj(obj)
            except ValidationError as e:
                # We are handling possibly misconfigured specs and
                # want to do a best-effort job to get a reasonable interface out of it.
                new_obj = copy.deepcopy(obj)
                for error in e.errors():
                    keys = error["loc"]
                    item = new_obj
                    for key in keys[:-1]:
                        item = item[key]
                    item.pop(keys[-1], None)
                return cls.parse_obj(new_obj)

        @classmethod
        def from_spec_dict(cls, spec_dict: dict) -> OpenAPISpec:
            """Get an OpenAPI spec from a dict."""
            return cls.parse_obj(spec_dict)

        @classmethod
        def from_text(cls, text: str) -> OpenAPISpec:
            """Get an OpenAPI spec from a text."""
            try:
                spec_dict = json.loads(text)
            except json.JSONDecodeError:
                spec_dict = yaml.safe_load(text)
            return cls.from_spec_dict(spec_dict)

        @classmethod
        def from_file(cls, path: Union[str, Path]) -> OpenAPISpec:
            """Get an OpenAPI spec from a file path."""
            path_ = path if isinstance(path, Path) else Path(path)
            if not path_.exists():
                raise FileNotFoundError(f"{path} does not exist")
            with path_.open("r") as f:
                return cls.from_text(f.read())

        @classmethod
        def from_url(cls, url: str) -> OpenAPISpec:
            """Get an OpenAPI spec from a URL."""
            response = requests.get(url)
            return cls.from_text(response.text)

        @property
        def base_url(self) -> str:
            """Get the base url."""
            return self.servers[0].url

        def get_methods_for_path(self, path: str) -> List[str]:
            """Return a list of valid methods for the specified path."""
            path_item = self._get_path_strict(path)
            results = []
            for method in HTTPVerb:
                operation = getattr(path_item, method.value, None)
                if isinstance(operation, Operation):
                    results.append(method.value)
            return results

        def get_parameters_for_path(self, path: str) -> List[Parameter]:
            path_item = self._get_path_strict(path)
            parameters = []
            if not path_item.parameters:
                return []
            for parameter in path_item.parameters:
                if isinstance(parameter, Reference):
                    parameter = self._get_root_referenced_parameter(parameter)
                parameters.append(parameter)
            return parameters

        def get_operation(self, path: str, method: str) -> Operation:
            """Get the operation object for a given path and HTTP method."""
            path_item = self._get_path_strict(path)
            operation_obj = getattr(path_item, method, None)
            if not isinstance(operation_obj, Operation):
                raise ValueError(f"No {method} method found for {path}")
            return operation_obj

        def get_parameters_for_operation(self, operation: Operation) -> List[Parameter]:
            """Get the components for a given operation."""
            parameters = []
            if operation.parameters:
                for parameter in operation.parameters:
                    if isinstance(parameter, Reference):
                        parameter = self._get_root_referenced_parameter(parameter)
                    parameters.append(parameter)
            return parameters

        def get_request_body_for_operation(
            self, operation: Operation
        ) -> Optional[RequestBody]:
            """Get the request body for a given operation."""
            request_body = operation.requestBody
            if isinstance(request_body, Reference):
                request_body = self._get_root_referenced_request_body(request_body)
            return request_body

        @staticmethod
        def get_cleaned_operation_id(
            operation: Operation, path: str, method: str
        ) -> str:
            """Get a cleaned operation id from an operation id."""
            operation_id = operation.operationId
            if operation_id is None:
                # Replace all punctuation of any kind with underscore
                path = re.sub(r"[^a-zA-Z0-9]", "_", path.lstrip("/"))
                operation_id = f"{path}_{method}"
            return operation_id.replace("-", "_").replace(".", "_").replace("/", "_")

else:

    class OpenAPISpec:  # type: ignore[no-redef]
        """Shim for pydantic version >=2"""

        def __init__(self) -> None:
            raise NotImplementedError("Only supported for pydantic version 1")
