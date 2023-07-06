from typing import Any
from langchain.chains.openai_functions.utils import _convert_schema, get_llm_kwargs
import json

def from_pydantic_create_openai_function_dict(
        pydantic_schema: Any, func_name: str, func_description: str
    ) -> dict:
    """
    Converts a pydantic schema and func property to a dict that is a correct function calling format 
    for the llm_kwargs prop for LLMChain

    Args:
    pydantic_schema: pydantic model
    func_name: str
    func_description: str

    Returns:
    The kwargs for the LLMChain constructor.

    """
    if not hasattr(pydantic_schema, "schema"):
        raise TypeError("pydantic_schema must be a pydantic model")

    function_props = {
        "name": func_name,
        "description": func_description,
        "parameters": _convert_schema(pydantic_schema.schema()),
    }

    llm_kwargs = get_llm_kwargs(function_props)

    return llm_kwargs


def from_json_create_openai_function_dict(
        json_schema: str, func_name: str, func_description: str
    ) -> dict:
    """
    Converts a json schema and func property to a dict that is a correct function calling format
    for the llm_kwargs prop for LLMChain

    Args:
    json_schema: json schema of output
    func_name: str
    func_description: str

    Returns:
    The kwargs for the LLMChain constructor.
    """

    try:
        json_dict = json.loads(json_schema)
    except:
        raise json.JSONDecodeError("json_schema must be a valid json string")
    json_dict["type"] = "object"
    function_props = {
        "name": func_name,
        "description": func_description,
        "parameters": json_dict,
    }

    llm_kwargs = get_llm_kwargs(function_props)

    return llm_kwargs


def from_dict_create_openai_function_dict(
        dict_schema: dict, func_name: str, func_description: str
    ) -> dict:
    """
    Converts a dictionary and func property to a dict that is a correct function calling format
    for the llm_kwargs prop for LLMChain

    Args:
    dict_schema: dictionary schema of output
    func_name: str
    func_description: str

    Returns:
    The kwargs for the LLMChain constructor.
    """
    if not isinstance(dict_schema, dict):
        raise TypeError("dict_schema must be a dictionary")
    dict_schema["type"] = "object"
    function_props = {
        "name": func_name,
        "description": func_description,
        "parameters": dict_schema,
    }

    llm_kwargs = get_llm_kwargs(function_props)

    return llm_kwargs