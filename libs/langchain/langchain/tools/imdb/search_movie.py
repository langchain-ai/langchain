from typing import Optional
import json

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool
from langchain.tools.imdb.utils import movies_to_dicts


class IMDbSearchMovie(IMDbBaseTool):
    """Tool that searches movies with a given title."""

    name: str = "imdb_search_movie"
    description: str = (
        "Searches IMDb for a movie with the given title and returns a "
        "JSON array containing the search results."
        "Useful for getting the ID number of a movie, given its title. "
        "The movies listed first are most relevant to the search."
    )

    def _run(
        self,
        title: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        movies = self.client.search_movie(title, results=20)
        return json.dumps(movies_to_dicts(movies))