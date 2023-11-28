from typing import Optional
import json

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.imdb.base import IMDbBaseTool
from langchain.tools.imdb.utils import people_to_dicts


class IMDbSearchPerson(IMDbBaseTool):
    """Tool that searches people with a given name."""

    name: str = "imdb_search_person"
    description: str = (
        "Searches IMDb for people with the given name and returns a "
        "JSON array containing the search results."
        "Useful for getting the ID number of a person, given their name. "
        "The people listed first are most relevant to the search."
    )

    def _run(
        self,
        name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        people = self.client.search_person(name, results=20)
        return json.dumps(people_to_dicts(people))