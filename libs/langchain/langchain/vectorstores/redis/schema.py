
from typing import List, Optional, Union
from typing_extensions import Literal
from pydantic import BaseModel, Field, validator

from redis.commands.search.field import (
    TextField,
    VectorField,
    TagField,
    GeoField,
    NumericField
)

class BaseField(BaseModel):
    name: str = Field(...)
    sortable: Optional[bool] = False


class TextFieldSchema(BaseField):
    weight: Optional[float] = 1
    no_stem: Optional[bool] = False
    phonetic_matcher: Optional[str] = None
    withsuffixtrie: Optional[bool] = False

    def as_field(self):
        return TextField(
            self.name,
            weight=self.weight,
            no_stem=self.no_stem,
            phonetic_matcher=self.phonetic_matcher,
            sortable=self.sortable,
        )


class TagFieldSchema(BaseField):
    separator: Optional[str] = ","
    case_sensitive: Optional[bool] = False

    def as_field(self):
        return TagField(
            self.name,
            separator=self.separator,
            case_sensitive=self.case_sensitive,
            sortable=self.sortable,
        )


class NumericFieldSchema(BaseField):
    def as_field(self):
        return NumericField(self.name, sortable=self.sortable)


class GeoFieldSchema(BaseField):
    def as_field(self):
        return GeoField(self.name, sortable=self.sortable)


class BaseVectorField(BaseModel):
    name: str = Field(...)
    dims: int = Field(...)
    algorithm: object = Field(...)
    datatype: str = Field(default="FLOAT32")
    distance_metric: str = Field(default="COSINE")
    initial_cap: int = Field(default=20000)

    @validator("algorithm", "datatype", "distance_metric")
    @classmethod
    def uppercase_strings(cls, v):
        return v.upper()

    @property
    def metric(self):
        return self.distance_metric


class FlatVectorField(BaseVectorField):
    algorithm: object = Literal["FLAT"]
    block_size: int = Field(default=1000)

    def as_field(self):
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


class HNSWVectorField(BaseVectorField):
    algorithm: object = Literal["HNSW"]
    m: int = Field(default=16)
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    epsilon: float = Field(default=0.8)

    def as_field(self):
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


class RedisMetadata(BaseModel):
    tag: Optional[List[TagFieldSchema]] = None
    text: Optional[List[TextFieldSchema]] = None
    numeric: Optional[List[NumericFieldSchema]] = None
    geo: Optional[List[GeoFieldSchema]] = None
    vector: Optional[List[Union[FlatVectorField, HNSWVectorField]]] = None

    def get_fields(self):
        redis_fields = []
        for field_name in self.__fields__.keys():
            field_group = getattr(self, field_name)
            if field_group is not None:
                for field in field_group:
                    redis_fields.append(field.as_field())
        return redis_fields

    @property
    def keys(self):
        keys = []
        for field_name in self.__fields__.keys():
            field_group = getattr(self, field_name)
            if field_group is not None:
                for field in field_group:
                    keys.append(field.name)
        return keys
