from typing import Any, Dict

from langchain.pydantic_v1 import root_validator
from langchain.tools.base import BaseTool


class IMDbBaseTool(BaseTool):
    """Base tool for IMDb."""

    client: Any = None  #: :meta private:

    description: str = (
        "A wrapper around IMDB movie Search. "
        "Useful for when you need to answer questions about movie, actor"
        "Input should be a search query."
    )

    @root_validator
    def validate_environment(cls, values: Dict) -> Dict:
        """A wrapper around IMDB movie Search.
        Useful for when you need to answer questions about movie, actor
        Input should be a search query."""
        try:
            from imdb import Cinemagoer
        except ImportError:
            raise ImportError(
                "Could not import the cinemagoer package. "
                "Please install it with `pip install git+https://github.com/cinemagoer/cinemagoer`."
            )
        values["client"] = Cinemagoer()
        return values
