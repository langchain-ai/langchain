import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool
from langchain.tools.imdb.utils import movies_to_dicts


class IMDbGetPersonMovies(IMDbBaseTool):
    """Tool that fetches movies that a person directed or acted in."""

    name: str = "imdb_get_person_movies"
    description: str = (
        "A wrapper around IMDb. "
        "Useful for when you need to know which movies a person has "
        "directed or acted in. Only the most recent movies are returned. "
        "Input should be a person ID number. "
        "Output is a JSON object containing the results."
    )

    def _run(
        self,
        id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from imdb import IMDbError

        try:
            person = self.client.get_person(id)
        except IMDbError:
            return (
                "The person could not be found. "
                "Please make sure to give a person ID instead of the person's name."
            )

        movies = {}
        acted_in = person.get("actor") or person.get("actress")
        if acted_in:
            movies["acted in"] = movies_to_dicts(acted_in[:20])
        directed = person.get("director")
        if directed:
            movies["directed"] = movies_to_dicts(directed[:20])
        return json.dumps(movies)
