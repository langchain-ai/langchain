from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel, Field, validator
from redis.commands.search.field import (
    GeoField,
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from typing_extensions import Literal

if TYPE_CHECKING:
    from redis.commands.search.field import Field as RedisField


class BaseField(BaseModel):
    name: str = Field(...)
    sortable: Optional[bool] = False


class TextFieldSchema(BaseField):
    weight: Optional[float] = 1
    no_stem: Optional[bool] = False
    phonetic_matcher: Optional[str] = None
    withsuffixtrie: Optional[bool] = False

    def as_field(self) -> TextField:
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

    def as_field(self) -> TagField:
        return TagField(
            self.name,
            separator=self.separator,
            case_sensitive=self.case_sensitive,
            sortable=self.sortable,
        )


class NumericFieldSchema(BaseField):
    def as_field(self) -> NumericField:
        return NumericField(self.name, sortable=self.sortable)


class GeoFieldSchema(BaseField):
    def as_field(self) -> GeoField:
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
    def uppercase_strings(cls, v: str) -> str:
        return v.upper()

    @property
    def metric(self) -> str:
        return self.distance_metric


class FlatVectorField(BaseVectorField):
    algorithm: object = Literal["FLAT"]
    block_size: int = Field(default=1000)

    def as_field(self) -> VectorField:
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

    def as_field(self) -> VectorField:
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

    @property
    def is_empty(self) -> bool:
        return all(
            field is None
            for field in [self.tag, self.text, self.numeric, self.geo, self.vector]
        )

    def get_fields(self) -> List["RedisField"]:
        redis_fields: List["RedisField"] = []
        if self.is_empty:
            return redis_fields

        for field_name in self.__fields__.keys():
            field_group = getattr(self, field_name)
            if field_group is not None:
                for field in field_group:
                    redis_fields.append(field.as_field())
        return redis_fields

    @property
    def keys(self) -> List[str]:
        keys: List[str] = []
        if self.is_empty:
            return keys

        for field_name in self.__fields__.keys():
            field_group = getattr(self, field_name)
            if field_group is not None:
                for field in field_group:
                    keys.append(field.name)
        return keys
