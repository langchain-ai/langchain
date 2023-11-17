import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests

from langchain.pydantic_v1 import BaseModel, Field, SecretStr


class StrEnum(str, Enum):
    """StrEnum string-based enum"""


class Operator(StrEnum):
    equals = "equals"
    notEquals = "notEquals"
    contains = "contains"
    notContains = "notContains"
    startsWith = "startsWith"
    endsWith = "endsWith"
    gt = "gt"
    gte = "gte"
    lt = "lt"
    lte = "lte"
    set = "set"
    notSet = "notSet"
    inDateRange = "inDateRange"
    notInDateRange = "notInDateRange"
    beforeDate = "beforeDate"
    afterDate = "afterDate"
    measureFilter = "measureFilter"


class Order(StrEnum):
    asc = "asc"
    desc = "desc"


class Granularity(StrEnum):
    second = "second"
    minute = "minute"
    hour = "hour"
    day = "day"
    week = "week"
    month = "month"
    quarter = "quarter"
    year = "year"


class Filter(BaseModel):
    member: str = Field(
        description="dimension or measure column",
    )
    operator: Operator = Field(...)
    values: List[str] = Field(...)


class TimeDimension(BaseModel):
    dimension: str = Field(description="dimension column")
    dateRange: List[Union[datetime, date]] = Field(
        min_items=2,
        max_items=2,
        description="An array of dates with the following format YYYY-MM-DD"
        " or in YYYY-MM-DDTHH:mm:ss.SSS format.",
    )
    granularity: Optional[Granularity] = Field(
        default=None,
        description="A granularity for a time dimension. If you pass null to the"
        " granularity, Cube will only perform filtering by a specified"
        " time dimension, without grouping.",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }


class Query(BaseModel):
    measures: Optional[List[str]] = Field(
        description="measure columns",
        default=None,
    )
    dimensions: Optional[List[str]] = Field(
        description="dimension columns",
        default=None,
    )
    filters: Optional[List[Filter]] = Field(
        default=None,
    )
    timeDimensions: Optional[List[TimeDimension]] = Field(
        default=None,
    )
    limit: Optional[int] = Field(
        default=None,
    )
    offset: Optional[int] = Field(
        default=None,
    )
    order: Optional[Dict[str, Order]] = Field(
        default=None,
        description="The keys are measures columns or dimensions columns to order by.",
    )


class Cube:
    """Cube Client.

    *Security Note*: This Cube Client interacts with an external service.

        Control access to who can use this Cube Client.

        Make sure that the capabilities given by this Cube Client to the calling
        code are appropriately scoped to the application.

        See https://python.langchain.com/docs/security or https://cube.dev/security
         for more information.
    """

    def __init__(
        self,
        cube_api_url: str,
        cube_api_token: SecretStr,
        *,
        ignore_models: Optional[List[str]] = None,
        include_models: Optional[List[str]] = None,
        custom_model_info: Optional[dict] = None,
    ):
        self.cube_api_url = cube_api_url
        self.cube_api_token = cube_api_token
        self._meta_information = self._get_meta_information()
        self._all_models = set([model["name"] for model in self._meta_information])

        self._include_models = set(include_models) if include_models else set()
        if self._include_models:
            missing_models = self._include_models - self._all_models
            if missing_models:
                raise ValueError(f"include_models {missing_models} not found in cube")

        self._ignore_models = set(ignore_models) if ignore_models else set()
        if self._ignore_models:
            missing_models = self._ignore_models - self._all_models
            if missing_models:
                raise ValueError(f"ignore_models {missing_models} not found in cube")

        usable_models = self.get_usable_model_names()
        self._usable_models = set(usable_models) if usable_models else self._all_models

        self._custom_model_info = custom_model_info
        if self._custom_model_info:
            if not isinstance(self._custom_model_info, dict):
                raise TypeError(
                    "model_info must be a dictionary with model names as keys and the "
                    "desired model info as values"
                )
            # only keep the models that are also present in the database
            intersection = set(self._custom_model_info).intersection(self._all_models)
            self._custom_model_info = dict(
                (model, self._custom_model_info[model])
                for model in self._custom_model_info
                if model in intersection
            )

    @property
    def meta_information(self) -> List[Dict[str, Any]]:
        """Metadata of the cube."""
        return self._meta_information

    def _get_meta_information(self) -> List[Dict[str, Any]]:
        """Metadata of the cube.
        See https://cube.dev/docs/reference/rest-api#v1meta for more information.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.cube_api_token.get_secret_value(),
        }

        response = requests.get(f"{self.cube_api_url}/meta", headers=headers)
        response.raise_for_status()
        raw_meta = response.json()
        cubes = raw_meta.get("cubes", [])

        if not cubes:
            raise ValueError("No cubes found in metadata.")

        return cubes

    def get_usable_model_names(self) -> List[str]:
        """Get names of models available."""
        if self._include_models:
            return sorted(self._include_models)
        return sorted(self._all_models - self._ignore_models)

    def get_usable_models(self) -> str:
        """Get available models."""

        all_model_names = self.get_usable_model_names()

        model_info = "| Model | Description |\n"
        model_info += "| --- | --- |\n"

        for m in self._meta_information:
            model_name = m["name"]
            if model_name not in all_model_names:
                continue

            description = m["title"]
            if m.get("description"):
                description += f", {m['description']}"

            model_info += f"| {model_name} | {description} |\n"

        return model_info

    @property
    def model_meta_information(self) -> str:
        """Information about all models in the cube."""
        return self.get_model_meta_information()

    def get_model_meta_information(
        self, model_names: Optional[List[str]] = None
    ) -> str:
        """Get information about specified models."""

        all_model_names = self.get_usable_model_names()
        if model_names is not None:
            missing_models = set(model_names).difference(all_model_names)
            if missing_models:
                raise ValueError(f"model_names {missing_models} not found in cube")
            all_model_names = model_names

        models = []
        for m in self._meta_information:
            model_name = m["name"]

            if model_name not in all_model_names:
                continue

            if self._custom_model_info and model_name in self._custom_model_info:
                models.append(self._custom_model_info[model_name])
                continue
            information = f"## Model: {model_name}\n"

            if m["measures"]:
                information += "\n### Measures:\n"
                information += "| Title | Description | Column | Type |\n"
                information += "| --- | --- | --- | --- |\n"

                for measure in m["measures"]:
                    information += (
                        f"| {measure.get('shortTitle')} "
                        f"| {measure.get('description', '')} "
                        f"| {measure.get('name')} "
                        f"| {measure.get('type')} |\n"
                    )

            if m["dimensions"]:
                information += "\n### Dimensions:\n"
                information += "| Title | Description | Column | Type |\n"
                information += "| --- | --- | --- | --- |\n"

                for dimension in m["dimensions"]:
                    information += (
                        f"| {dimension.get('shortTitle')} "
                        f"| {dimension.get('description', '')} "
                        f"| {dimension.get('name')} "
                        f"| {dimension.get('type')} |\n"
                    )

            models.append(information)

        models.sort()

        return "\n\n".join(models)

    def load(self, query: Union[Dict[str, Any], str, Query]) -> Dict[str, Any]:
        """Load of the cube.
        See https://cube.dev/docs/reference/rest-api#v1load for more information.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.cube_api_token.get_secret_value(),
        }

        params = {"query": ""}

        if isinstance(query, Query):
            params["query"] = query.json(exclude_unset=True)
        elif isinstance(query, Dict):
            params["query"] = json.dumps(query)

        response = requests.post(
            url=f"{self.cube_api_url}/load",
            headers=headers,
            json=params,
        )

        raw = response.json()
        if "error" in raw:
            raise ValueError(raw["error"])
        else:
            response.raise_for_status()

        return raw

    def sql(self, query: Union[Dict[str, Any], Query]) -> Dict[str, Any]:
        """SQL of the cube.
        See https://cube.dev/docs/reference/rest-api#v1sql for more information.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.cube_api_token.get_secret_value(),
        }

        params = {"query": ""}

        if isinstance(query, Query):
            params["query"] = query.json(exclude_unset=True)
        elif isinstance(query, Dict):
            params["query"] = json.dumps(query)

        response = requests.get(
            url=f"{self.cube_api_url}/sql",
            headers=headers,
            params=params,
        )

        raw = response.json()
        if "error" in raw:
            raise ValueError(raw["error"])
        else:
            response.raise_for_status()

        return raw["sql"]
