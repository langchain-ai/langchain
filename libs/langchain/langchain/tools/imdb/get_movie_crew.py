from typing import Optional
import json

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool
from langchain.tools.imdb.utils import people_to_dicts


class IMDbGetMovieCrew(IMDbBaseTool):
    """Tool that fetches the crew of a movie."""

    name: str = "imdb_get_movie_crew"
    description: str = (
        "A wrapper around IMDb. "
        "Useful for when you need to know who the directors, writers, "
        "producers, cinematographers, editors, or composers "
        "for a movie are."
        "Input should be a movie ID number. "
        "Output is a JSON object containing the results."
    )

    def _run(
        self,
        id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from imdb import IMDbError
        try:
            movie = self.client.get_movie(id)
        except IMDbError:
            return (
                "The movie could not be found. "
                "Please make sure to give a movie ID instead of the movie title."
            )

        keys = [
            'director',
            'writer',
            'producer',
            'cinematographer',
            'editor',
            'composer',
        ]
        crew = {
            key: people_to_dicts(movie.get(key))
            for key in keys
            if key in movie
        }
        return json.dumps(crew)