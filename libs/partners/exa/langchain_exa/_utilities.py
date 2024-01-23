import os
from typing import Dict

from exa_py import Exa  # type: ignore
from langchain_core.utils import convert_to_secret_str


def initialize_client(values: Dict) -> Dict:
    """Initialize the client."""
    exa_api_key = values.get("exa_api_key") or os.environ.get("EXA_API_KEY") or ""
    values["exa_api_key"] = convert_to_secret_str(exa_api_key)
    values["_client"] = Exa(api_key=values["exa_api_key"].get_secret_value())
    return values
