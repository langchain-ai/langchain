from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool


class IMDbGetPersonBio(IMDbBaseTool):
    """Tool that fetches the biography of a person from IMDb."""

    name: str = "imdb_get_person_bio"
    description: str = (
        "A wrapper around IMDb. "
        "Useful for when you need a biography of an actor, director, "
        "or someone else that has worked on a movie."
        "Input should be a person ID number."
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
        
        bio = person.get('mini biography')
        if bio:
            return bio[0]
        return "This person does not have a biography."