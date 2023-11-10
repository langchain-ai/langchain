"""
Adapted from https://github.com/venuv/langchain_yt_tools

CustomYTSearchTool searches YouTube videos related to a person
and returns a specified number of video URLs.
Input to this tool should be a comma separated list,
 - the first part contains a person name
 - and the second(optional) a number that is the
    maximum number of video results to return
 """
import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool


class YouTubeSearchTool(BaseTool):
    """Tool that queries YouTube."""

    name: str = "youtube_search"
    description: str = (
        "search for youtube videos associated with a topic or a person. "
        "the input to this tool should be a comma separated list, "
        "the first part contains the name of a topic or person name "
        "and the second a number that is the maximum number of video "
        "results  to return aka num_results. the second part is optional"
    )

    def _search(self, query: str, num_results: int) -> str:
        from youtube_search import YoutubeSearch

        results = YoutubeSearch(query, num_results).to_json()
        data = json.loads(results)
        url_suffix_list = [
            "https://www.youtube.com" + video["url_suffix"] for video in data["videos"]
        ]
        return str(url_suffix_list)

    def _run(
        self,
        query: str,
        num_results: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        values = query.split(",")
        query = values[0]
        if not num_results and len(values) > 1:
            # if num_results was passed in the query string
            num_results = int(values[1])
        elif not num_results:
            # if num_results was not provided
            num_results = 2

        return self._search(query, num_results)
