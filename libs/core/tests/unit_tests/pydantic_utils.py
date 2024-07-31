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

    if "properties" in schema_:
        properties = schema_["properties"]
        if "configurable" in properties:
            configurable = properties["configurable"]

            if "allOf" in configurable:
                allOf = configurable["allOf"]
                del configurable["allOf"]
                configurable["$ref"] = allOf[0]["$ref"]
                if "default" in configurable:
                    del configurable["default"]

    if "allOf" in schema_:
        allOf = schema_["allOf"]
        del schema_["allOf"]
        schema_["$ref"] = allOf[0]["$ref"]
    return schema_
