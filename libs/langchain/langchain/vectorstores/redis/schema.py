from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from typing_extensions import TYPE_CHECKING, Literal

from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.vectorstores.redis.constants import REDIS_VECTOR_DTYPE_MAP

if TYPE_CHECKING:
    from redis.commands.search.field import (  # type: ignore
        NumericField,
        TagField,
        TextField,
        VectorField,
    )


class RedisDistanceMetric(str, Enum):
    """Distance metrics for Redis vector fields."""

    l2 = "L2"
    cosine = "COSINE"
    ip = "IP"


class RedisField(BaseModel):
    """Base class for Redis fields."""

    name: str = Field(...)


class TextFieldSchema(RedisField):
    """Schema for text fields in Redis."""

    weight: float = 1
    no_stem: bool = False
    phonetic_matcher: Optional[str] = None
    withsuffixtrie: bool = False
    no_index: bool = False
    sortable: Optional[bool] = False

    def as_field(self) -> TextField:
        from redis.commands.search.field import TextField  # type: ignore

        return TextField(
            self.name,
            weight=self.weight,
            no_stem=self.no_stem,
            phonetic_matcher=self.phonetic_matcher,
            sortable=self.sortable,
            no_index=self.no_index,
        )


class TagFieldSchema(RedisField):
    """Schema for tag fields in Redis."""

    separator: str = ","
    case_sensitive: bool = False
    no_index: bool = False
    sortable: Optional[bool] = False

    def as_field(self) -> TagField:
        from redis.commands.search.field import TagField  # type: ignore

        return TagField(
            self.name,
            separator=self.separator,
            case_sensitive=self.case_sensitive,
            sortable=self.sortable,
            no_index=self.no_index,
        )


class NumericFieldSchema(RedisField):
    """Schema for numeric fields in Redis."""

    no_index: bool = False
    sortable: Optional[bool] = False

    def as_field(self) -> NumericField:
        from redis.commands.search.field import NumericField  # type: ignore

        return NumericField(self.name, sortable=self.sortable, no_index=self.no_index)


class RedisVectorField(RedisField):
    """Base class for Redis vector fields."""

    dims: int = Field(...)
    algorithm: object = Field(...)
    datatype: str = Field(default="FLOAT32")
    distance_metric: RedisDistanceMetric = Field(default="COSINE")
    initial_cap: int = Field(default=20000)

    @validator("distance_metric", pre=True)
    def uppercase_strings(cls, v: str) -> str:
        return v.upper()

    @validator("datatype", pre=True)
    def uppercase_and_check_dtype(cls, v: str) -> str:
        if v.upper() not in REDIS_VECTOR_DTYPE_MAP:
            raise ValueError(
                f"datatype must be one of {REDIS_VECTOR_DTYPE_MAP.keys()}. Got {v}"
            )
        return v.upper()


class FlatVectorField(RedisVectorField):
    """Schema for flat vector fields in Redis."""

    algorithm: Literal["FLAT"] = "FLAT"
    block_size: int = Field(default=1000)

    def as_field(self) -> VectorField:
        from redis.commands.search.field import VectorField  # type: ignore

        return VectorField(
            self.name,
            self.algorithm,
            {
                "TYPE": self.datatype,
                "DIM": self.dims,
                "DISTANCE_METRIC": self.distance_metric,
                "INITIAL_CAP": self.initial_cap,
                "BLOCK_SIZE": self.block_size,
            },
        )


class HNSWVectorField(RedisVectorField):
    """Schema for HNSW vector fields in Redis."""

    algorithm: Literal["HNSW"] = "HNSW"
    m: int = Field(default=16)
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    epsilon: float = Field(default=0.8)

    def as_field(self) -> VectorField:
        from redis.commands.search.field import VectorField  # type: ignore

        return VectorField(
            self.name,
            self.algorithm,
            {
                "TYPE": self.datatype,
                "DIM": self.dims,
                "DISTANCE_METRIC": self.distance_metric,
                "INITIAL_CAP": self.initial_cap,
                "M": self.m,
                "EF_CONSTRUCTION": self.ef_construction,
                "EF_RUNTIME": self.ef_runtime,
                "EPSILON": self.epsilon,
            },
        )


class RedisModel(BaseModel):
    """Schema for Redis index."""

    # always have a content field for text
    text: List[TextFieldSchema] = [TextFieldSchema(name="content")]
    tag: Optional[List[TagFieldSchema]] = None
    numeric: Optional[List[NumericFieldSchema]] = None
    extra: Optional[List[RedisField]] = None

    # filled by default_vector_schema
    vector: Optional[List[Union[FlatVectorField, HNSWVectorField]]] = None
    content_key: str = "content"
    content_vector_key: str = "content_vector"

    def add_content_field(self) -> None:
        if self.text is None:
            self.text = []
        for field in self.text:
            if field.name == self.content_key:
                return
        self.text.append(TextFieldSchema(name=self.content_key))

    def add_vector_field(self, vector_field: Dict[str, Any]) -> None:
        # catch case where user inputted no vector field spec
        # in the index schema
        if self.vector is None:
            self.vector = []

        # ignore types as pydantic is handling type validation and conversion
        if vector_field["algorithm"] == "FLAT":
            self.vector.append(FlatVectorField(**vector_field))  # type: ignore
        elif vector_field["algorithm"] == "HNSW":
            self.vector.append(HNSWVectorField(**vector_field))  # type: ignore
        else:
            raise ValueError(
                f"algorithm must be either FLAT or HNSW. Got "
                f"{vector_field['algorithm']}"
            )

    def as_dict(self) -> Dict[str, List[Any]]:
        schemas: Dict[str, List[Any]] = {"text": [], "tag": [], "numeric": []}
        # iter over all class attributes
        for attr, attr_value in self.__dict__.items():
            # only non-empty lists
            if isinstance(attr_value, list) and len(attr_value) > 0:
                field_values: List[Dict[str, Any]] = []
                # iterate over all fields in each category (tag, text, etc)
                for val in attr_value:
                    value: Dict[str, Any] = {}
                    # iterate over values within each field to extract
                    # settings for that field (i.e. name, weight, etc)
                    for field, field_value in val.__dict__.items():
                        # make enums into strings
                        if isinstance(field_value, Enum):
                            value[field] = field_value.value
                        # don't write null values
                        elif field_value is not None:
                            value[field] = field_value
                    field_values.append(value)

                schemas[attr] = field_values

        schema: Dict[str, List[Any]] = {}
        # only write non-empty lists from defaults
        for k, v in schemas.items():
            if len(v) > 0:
                schema[k] = v
        return schema

    @property
    def content_vector(self) -> Union[FlatVectorField, HNSWVectorField]:
        if not self.vector:
            raise ValueError("No vector fields found")
        for field in self.vector:
            if field.name == self.content_vector_key:
                return field
        raise ValueError("No content_vector field found")

    @property
    def vector_dtype(self) -> np.dtype:
        # should only ever be called after pydantic has validated the schema
        return REDIS_VECTOR_DTYPE_MAP[self.content_vector.datatype]

    @property
    def is_empty(self) -> bool:
        return all(
            field is None for field in [self.tag, self.text, self.numeric, self.vector]
        )

    def get_fields(self) -> List["RedisField"]:
        redis_fields: List["RedisField"] = []
        if self.is_empty:
            return redis_fields

        for field_name in self.__fields__.keys():
            if field_name not in ["content_key", "content_vector_key", "extra"]:
                field_group = getattr(self, field_name)
                if field_group is not None:
                    for field in field_group:
                        redis_fields.append(field.as_field())
        return redis_fields

    @property
    def metadata_keys(self) -> List[str]:
        keys: List[str] = []
        if self.is_empty:
            return keys

        for field_name in self.__fields__.keys():
            field_group = getattr(self, field_name)
            if field_group is not None:
                for field in field_group:
                    # check if it's a metadata field. exclude vector and content key
                    if not isinstance(field, str) and field.name not in [
                        self.content_key,
                        self.content_vector_key,
                    ]:
                        keys.append(field.name)
        return keys


def read_schema(
    index_schema: Optional[Union[Dict[str, str], str, os.PathLike]]
) -> Dict[str, Any]:
    """Reads in the index schema from a dict or yaml file.

    Check if it is a dict and return RedisModel otherwise, check if it's a path and
    read in the file assuming it's a yaml file and return a RedisModel
    """
    if isinstance(index_schema, dict):
        return index_schema
    elif isinstance(index_schema, Path):
        with open(index_schema, "rb") as f:
            return yaml.safe_load(f)
    elif isinstance(index_schema, str):
        if Path(index_schema).resolve().is_file():
            with open(index_schema, "rb") as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"index_schema file {index_schema} does not exist")
    else:
        raise TypeError(
            f"index_schema must be a dict, or path to a yaml file "
            f"Got {type(index_schema)}"
        )
