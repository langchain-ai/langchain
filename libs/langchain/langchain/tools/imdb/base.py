from typing import Any, Dict

from langchain.pydantic_v1 import root_validator
from langchain.tools.base import BaseTool


class IMDbBaseTool(BaseTool):
    """Base tool for IMDb."""

    client: Any = None  #: :meta private:

    @root_validator
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            from imdb import Cinemagoer
        except ImportError:
            raise ImportError(
                "Could not import the cinemagoer package. "
                "Please install it with `pip install git+https://github.com/cinemagoer/cinemagoer`."
            )
        values["client"] = Cinemagoer()
        return values
