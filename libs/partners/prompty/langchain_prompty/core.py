from __future__ import annotations

import abc
import json
import os
import re
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, FilePath

T = TypeVar("T")


class SimpleModel(BaseModel, Generic[T]):
    """Simple model for a single item."""

    item: T


class PropertySettings(BaseModel):
    """Property settings for a prompty model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["string", "number", "array", "object", "boolean"]
    default: str | int | float | list | dict | bool | None = Field(default=None)
    description: str = Field(default="")


class ModelSettings(BaseModel):
    """Model settings for a prompty model."""

    api: str = Field(default="")
    configuration: dict = Field(default={})
    parameters: dict = Field(default={})
    response: dict = Field(default={})

    def model_dump_safe(self) -> dict:
        d = self.model_dump()
        d["configuration"] = {
            k: "*" * len(v) if "key" in k.lower() or "secret" in k.lower() else v
            for k, v in d["configuration"].items()
        }
        return d


class TemplateSettings(BaseModel):
    """Template settings for a prompty model."""

    type: str = Field(default="mustache")
    parser: str = Field(default="")


class Prompty(BaseModel):
    """Base Prompty model."""

    # Metadata
    name: str = Field(default="")
    description: str = Field(default="")
    authors: list[str] = Field(default=[])
    tags: list[str] = Field(default=[])
    version: str = Field(default="")
    base: str = Field(default="")
    basePrompty: Prompty | None = Field(default=None)

    # Model
    model: ModelSettings = Field(default_factory=ModelSettings)

    # Sample
    sample: dict = Field(default={})

    # Input / output
    inputs: dict[str, PropertySettings] = Field(default={})
    outputs: dict[str, PropertySettings] = Field(default={})

    # Template
    template: TemplateSettings

    file: FilePath = Field(default="")  # type: ignore[assignment]
    content: str = Field(default="")

    def to_safe_dict(self) -> dict[str, Any]:
        d = {}
        for k, v in self:
            if v != "" and v != {} and v != [] and v is not None:
                if k == "model":
                    d[k] = v.model_dump_safe()
                elif k == "template":
                    d[k] = v.model_dump()
                elif k == "inputs" or k == "outputs":
                    d[k] = {k: v.model_dump() for k, v in v.items()}
                elif k == "file":
                    d[k] = (
                        str(self.file.as_posix())
                        if isinstance(self.file, Path)
                        else self.file
                    )
                elif k == "basePrompty":
                    # No need to serialize basePrompty
                    continue

                else:
                    d[k] = v
        return d

    # Generate json representation of the prompty
    def to_safe_json(self) -> str:
        d = self.to_safe_dict()
        return json.dumps(d)

    @staticmethod
    def normalize(attribute: Any, parent: Path, env_error: bool = True) -> Any:
        if isinstance(attribute, str):
            attribute = attribute.strip()
            if attribute.startswith("${") and attribute.endswith("}"):
                variable = attribute[2:-1].split(":")
                if variable[0] in os.environ.keys():
                    return os.environ[variable[0]]
                else:
                    if len(variable) > 1:
                        return variable[1]
                    else:
                        if env_error:
                            raise ValueError(
                                f"Variable {variable[0]} not found in environment"
                            )
                        else:
                            return ""
            elif (
                attribute.startswith("file:")
                and Path(parent / attribute.split(":")[1]).exists()
            ):
                with open(parent / attribute.split(":")[1]) as f:
                    items = json.load(f)
                    if isinstance(items, list):
                        return [Prompty.normalize(value, parent) for value in items]
                    elif isinstance(items, dict):
                        return {
                            key: Prompty.normalize(value, parent)
                            for key, value in items.items()
                        }
                    else:
                        return items
            else:
                return attribute
        elif isinstance(attribute, list):
            return [Prompty.normalize(value, parent) for value in attribute]
        elif isinstance(attribute, dict):
            return {
                key: Prompty.normalize(value, parent)
                for key, value in attribute.items()
            }
        else:
            return attribute


def param_hoisting(
    top: dict[str, Any], bottom: dict[str, Any], top_key: Any = None
) -> dict[str, Any]:
    """Merge two dictionaries with hoisting of parameters from bottom to top.

    Args:
        top: The top dictionary.
        bottom: The bottom dictionary.
        top_key: The key to hoist from the bottom to the top.

    Returns:
        The merged dictionary.
    """
    if top_key:
        new_dict = {**top[top_key]} if top_key in top else {}
    else:
        new_dict = {**top}
    for key, value in bottom.items():
        if key not in new_dict:
            new_dict[key] = value
    return new_dict


class Invoker(abc.ABC):
    """Base class for all invokers."""

    def __init__(self, prompty: Prompty) -> None:
        self.prompty = prompty

    @abc.abstractmethod
    def invoke(self, data: BaseModel) -> BaseModel:
        pass

    def __call__(self, data: BaseModel) -> BaseModel:
        return self.invoke(data)


class NoOpParser(Invoker):
    """NoOp parser for invokers."""

    def invoke(self, data: BaseModel) -> BaseModel:
        return data


class InvokerFactory:
    """Factory for creating invokers."""

    _instance = None
    _renderers: dict[str, type[Invoker]] = {}
    _parsers: dict[str, type[Invoker]] = {}
    _executors: dict[str, type[Invoker]] = {}
    _processors: dict[str, type[Invoker]] = {}

    def __new__(cls) -> InvokerFactory:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Add NOOP invokers
            cls._renderers["NOOP"] = NoOpParser
            cls._parsers["NOOP"] = NoOpParser
            cls._executors["NOOP"] = NoOpParser
            cls._processors["NOOP"] = NoOpParser
        return cls._instance

    def register(
        self,
        type: Literal["renderer", "parser", "executor", "processor"],
        name: str,
        invoker: type[Invoker],
    ) -> None:
        if type == "renderer":
            self._renderers[name] = invoker
        elif type == "parser":
            self._parsers[name] = invoker
        elif type == "executor":
            self._executors[name] = invoker
        elif type == "processor":
            self._processors[name] = invoker
        else:
            raise ValueError(f"Invalid type {type}")

    def register_renderer(self, name: str, renderer_class: Any) -> None:
        self.register("renderer", name, renderer_class)

    def register_parser(self, name: str, parser_class: Any) -> None:
        self.register("parser", name, parser_class)

    def register_executor(self, name: str, executor_class: Any) -> None:
        self.register("executor", name, executor_class)

    def register_processor(self, name: str, processor_class: Any) -> None:
        self.register("processor", name, processor_class)

    def __call__(
        self,
        type: Literal["renderer", "parser", "executor", "processor"],
        name: str,
        prompty: Prompty,
        data: BaseModel,
    ) -> Any:
        if type == "renderer":
            return self._renderers[name](prompty)(data)
        elif type == "parser":
            return self._parsers[name](prompty)(data)
        elif type == "executor":
            return self._executors[name](prompty)(data)
        elif type == "processor":
            return self._processors[name](prompty)(data)
        else:
            raise ValueError(f"Invalid type {type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "renderers": {
                k: f"{v.__module__}.{v.__name__}" for k, v in self._renderers.items()
            },
            "parsers": {
                k: f"{v.__module__}.{v.__name__}" for k, v in self._parsers.items()
            },
            "executors": {
                k: f"{v.__module__}.{v.__name__}" for k, v in self._executors.items()
            },
            "processors": {
                k: f"{v.__module__}.{v.__name__}" for k, v in self._processors.items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class Frontmatter:
    """Class for reading frontmatter from a string or file."""

    _yaml_delim = r"(?:---|\+\+\+)"
    _yaml = r"(.*?)"
    _content = r"\s*(.+)$"
    _re_pattern = r"^\s*" + _yaml_delim + _yaml + _yaml_delim + _content
    _regex = re.compile(_re_pattern, re.S | re.M)

    @classmethod
    def read_file(cls, path: str) -> dict[str, Any]:
        """Reads file at path and returns dict with separated frontmatter.
        See read() for more info on dict return value.
        """
        with open(path, encoding="utf-8") as file:
            file_contents = file.read()
            return cls.read(file_contents)

    @classmethod
    def read(cls, string: str) -> dict[str, Any]:
        """Returns dict with separated frontmatter from string.

        Returned dict keys:
        - attributes: extracted YAML attributes in dict form.
        - body: string contents below the YAML separators
        - frontmatter: string representation of YAML
        """
        fmatter = ""
        body = ""
        result = cls._regex.search(string)

        if result:
            fmatter = result.group(1)
            body = result.group(2)
        return {
            "attributes": yaml.safe_load(fmatter),
            "body": body,
            "frontmatter": fmatter,
        }
