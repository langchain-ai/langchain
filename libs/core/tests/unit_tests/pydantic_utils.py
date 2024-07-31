from typing import Any, Type

from pydantic import BaseModel


def _schema(obj: Type[BaseModel]) -> dict:
    """Return the schema of the object."""
    # Remap to old style schema
    if not hasattr(obj, "model_json_schema"):  # V1 model
        return obj.schema()

    schema_ = obj.model_json_schema(ref_template="#/definitions/{model}")
    if "$defs" in schema_:
        schema_["definitions"] = schema_["$defs"]
        del schema_["$defs"]
    return schema_
