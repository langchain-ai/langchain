from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool


def people_to_dicts(people) -> dict:
    if not people:
        return people
    return [{"name": p.get("name"), "id": p.getID()} for p in people]


class IMDBCastOfMovie(IMDbBaseTool):
    """Tool to find cast of a movie given its name."""

    name: str = "imdb_get_movie_cast"
    description: str = """Use this tool to retrieve a list of cast members for a movie, 
    given its IMBD movie ID."""

    def _run(
        self, id: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        from imdb import IMDbError

        try:
            res_movie = self.client.get_movie(id)
        
        except IMDbError:
            return (
                "The movie could not be found. "
                "Please make sure to give a movie ID instead of a movie name."
            )


        return people_to_dicts(res_movie["cast"])
