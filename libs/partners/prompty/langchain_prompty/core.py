from __future__ import annotations

import os
import re
import yaml
import json
import abc
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, FilePath
from typing import List, Literal, Dict, Union, Generic, TypeVar

T = TypeVar("T")


class SimpleModel(BaseModel, Generic[T]):
    item: T


class PropertySettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["string", "number", "array", "object", "boolean"]
    default: str | int | float | List | dict | bool = Field(default=None)
    description: str = Field(default="")


class ModelSettings(BaseModel):
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
    type: str = Field(default="jinja2")
    parser: str = Field(default="")


class Prompty(BaseModel):
    # metadata
    name: str = Field(default="")
    description: str = Field(default="")
    authors: List[str] = Field(default=[])
    tags: List[str] = Field(default=[])
    version: str = Field(default="")
    base: str = Field(default="")
    basePrompty: Prompty | None = Field(default=None)
    # model
    model: ModelSettings = Field(default_factory=ModelSettings)

    # sample
    sample: dict = Field(default={})

    # input / output
    inputs: Dict[str, PropertySettings] = Field(default={})
    outputs: Dict[str, PropertySettings] = Field(default={})

    # template
    template: TemplateSettings

    file: FilePath = Field(default="")
    content: str | List[str] | dict = Field(default="")

    def to_safe_dict(self) -> Dict[str, any]:
        d = {}
        for k, v in self:
            if v != "" and v != {} and v != [] and v != None:
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
                    # no need to serialize basePrompty
                    continue

                else:
                    d[k] = v
        return d

    # generate json representation of the prompty
    def to_safe_json(self) -> str:
        d = self.to_safe_dict()
        return json.dumps(d)

    @staticmethod
    def normalize(attribute: any, parent: Path, env_error=True) -> any:
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
                with open(parent / attribute.split(":")[1], "r") as f:
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
    top: Dict[str, any], bottom: Dict[str, any], top_key: str = None
) -> Dict[str, any]:
    if top_key:
        new_dict = {**top[top_key]} if top_key in top else {}
    else:
        new_dict = {**top}
    for key, value in bottom.items():
        if not key in new_dict:
            new_dict[key] = value
    return new_dict


class Invoker(abc.ABC):
    def __init__(self, prompty: Prompty) -> None:
        self.prompty = prompty

    @abc.abstractmethod
    def invoke(self, data: BaseModel) -> BaseModel:
        pass

    def __call__(self, data: BaseModel) -> BaseModel:
        return self.invoke(data)


class NoOpParser(Invoker):
    def invoke(self, data: BaseModel) -> BaseModel:
        return data


class InvokerFactory(object):
    _instance = None 
    _renderers: Dict[str, Invoker] = {}
    _parsers: Dict[str, Invoker] = {}
    _executors: Dict[str, Invoker] = {}
    _processors: Dict[str, Invoker] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InvokerFactory, cls).__new__(cls)
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
        invoker: Invoker,
    ):
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

    def register_renderer(self, name, renderer_class):
        self.register("renderer", name, renderer_class)

    def register_parser(self, name, parser_class):
        self.register("parser", name, parser_class)

    def register_executor(self, name, executor_class):
        self.register("executor", name, executor_class)

    def register_processor(self, name, processor_class):
        self.register("processor", name, processor_class)

    def __call__(
        self,
        type: Literal["renderer", "parser", "executor", "processor"],
        name: str,
        prompty: Prompty,
        data: BaseModel,
    ) -> BaseModel:
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

    def to_dict(self) -> Dict[str, any]:
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
    _yaml_delim = r"(?:---|\+\+\+)"
    _yaml = r"(.*?)"
    _content = r"\s*(.+)$"
    _re_pattern = r"^\s*" + _yaml_delim + _yaml + _yaml_delim + _content
    _regex = re.compile(_re_pattern, re.S | re.M)

    @classmethod
    def read_file(cls, path):
        """Reads file at path and returns dict with separated frontmatter.
        See read() for more info on dict return value.
        """
        with open(path, encoding="utf-8") as file:
            file_contents = file.read()
            return cls.read(file_contents)

    @classmethod
    def read(cls, string):
        """Returns dict with separated frontmatter from string.

        Returned dict keys:
        attributes -- extracted YAML attributes in dict form.
        body -- string contents below the YAML separators
        frontmatter -- string representation of YAML
        """
        fmatter = ""
        body = ""
        result = cls._regex.search(string)

        if result:
            fmatter = result.group(1)
            body = result.group(2)
        return {
            "attributes": yaml.load(fmatter, Loader=yaml.FullLoader),
            "body": body,
            "frontmatter": fmatter,
        }
