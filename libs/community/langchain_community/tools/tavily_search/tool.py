"""Tool for the Tavily search API."""

import json
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


class TavilyInput(BaseModel):
    """Input for the Tavily tool."""

    query: str = Field(description="search query to look up")


class TavilySearchResults(BaseTool):
    """Tool that queries the Tavily Search API and gets back json.

    Setup:
        Install ``langchain-openai`` and ``tavily-python``, and set environment variable ``TAVILY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-openai
            export TAVILY_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_community.tools import TavilySearchResults

            tool = TavilySearchResults(
                version="v2",
                max_results=5,
                include_answer=True,
                include_raw_content=True,
                include_images=True,
                # search_depth="advanced",
                # include_domains = []
                # exclude_domains = []
            )

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({'query': 'who won the last french open'})

        .. code-block:: python

            '{\n  "answer": "Novak Djokovic won the last French Open by beating Casper Ruud in three sets (7-6(1), 6-3, 7-5) on Sunday, June 11, 2023.",\n  "results": [\n    {\n      "title": "Djokovic wins French Open, record 23rd Grand Slam title",\n      "url": "https://www.nytimes.com/athletic/4600616/2023/06/11/novak-djokovic-french-open-mens-final/",\n      "content": "Novak Djokovic beat Casper Ruud in three sets (7-6(1), 6-3, 7-5) Sunday to win the French Open men\'s final and capture his record-breaking 23rd Grand Slam title."\n    },\n    {\n      "title": "Novak Djokovic wins his 23rd Grand Slam title by beating Casper Ruud in ...",\n      "url": "https://apnews.com/article/djokovic-ruud-french-open-roland-garros-final-d7bda9f570b010ea48cf8a05b397291e",\n      "content": "Novak Djokovic wins his 23rd Grand Slam title by beating Casper Ruud in the French Open final\\nSerbia\\u2019s Novak Djokovic celebrates winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023. Serbia\\u2019s Novak Djokovic wears the number 23 on his garment for his 23rd Grand Slam Singles title as he kisses the trophy after winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023. Serbia\\u2019s Novak Djokovic wears the number 23 on his garment for his 23rd Grand Slam Singles title as he kisses the trophy after winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023. Serbia\\u2019s Novak Djokovic wears the number 23 on his garment for his 23rd Grand Slam Singles title as he kisses the trophy after winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023. Serbia\\u2019s Novak Djokovic kisses the trophy as he celebrates winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023."\n    },\n    {\n      "title": "Novak Djokovic wins his 23rd Grand Slam title : NPR",\n      "url": "https://www.npr.org/2023/06/11/1181568367/novak-djokovic-tennis-french-open-grand-slam",\n      "content": "Sports\\nNovak Djokovic wins the French Open men\'s singles, securing his 23rd Grand Slam title\\nBy\\nThe Associated Press\\nSerbia\'s Novak Djokovic celebrates winning the men\'s singles final match of the French Open tennis tournament against Norway\'s Casper Ruud in three sets at the Roland Garros stadium in Paris, Sunday.\\n Thibault Camus/AP\\nhide caption\\nSerbia\'s Novak Djokovic celebrates winning the men\'s singles final match of the French Open tennis tournament against Norway\'s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris on Sunday.\\n Thibault Camus/AP\\nhide caption\\nSerbia\'s Novak Djokovic celebrates winning the men\'s singles final match of the French Open tennis tournament against Norway\'s Casper Ruud in three sets at the Roland Garros stadium in Paris, Sunday.\\n At 20 days past his 36th birthday, Djokovic is the oldest singles champion at Roland Garros, considered the most grueling of the majors because of the lengthy, grinding points required by the red clay, which is slower than the grass or hard courts underfoot elsewhere.\\n Djokovic came close to pulling off that feat in 2021, when he won the Australian Open, French Open and Wimbledon and made it all the way to the title match at the U.S. Open before losing to Daniil Medvedev.\\n"\n    },\n    {\n      "title": "Winners of the Men\'s French Open - Topend Sports",\n      "url": "https://www.topendsports.com/events/tennis-grand-slam/french-open/winners-men.htm",\n      "content": "Here are all the winners of the French Tennis Open men\'s title since the first tournament in 1925. The recent tournaments have been dominated by Spanish player Rafael Nadal."\n    },\n    {\n      "title": "Carlos Alcaraz wins the French Open, earning a third Grand Slam title",\n      "url": "https://www.npr.org/2024/06/09/nx-s1-4997726/carlos-alcaraz-wins-french-open-third-grand-slam-title",\n      "content": "PARIS \\u2014 As Carlos Alcaraz began constructing his comeback in Sunday\'s French Open final, a 6-3, 2-6, 5-7, 6-1, 6-2 victory over Alexander Zverev for a first championship at Roland Garros and ..."\n    }\n  ]\n}',

    Invoke with tool call:

        .. code-block:: python

            tool.invoke({"args": {'query': 'who won the last french open'}, "type": "tool_call", "id": "foo", "name": "tavily"})

        .. code-block:: python

            ToolMessage(
                content='{\n  "answer": "Novak Djokovic won the last French Open by beating Casper Ruud in three sets (7-6(1), 6-3, 7-5) on Sunday, June 11, 2023.",\n  "results": [\n    {\n      "title": "Djokovic wins French Open, record 23rd Grand Slam title",\n      "url": "https://www.nytimes.com/athletic/4600616/2023/06/11/novak-djokovic-french-open-mens-final/",\n      "content": "Novak Djokovic beat Casper Ruud in three sets (7-6(1), 6-3, 7-5) Sunday to win the French Open men\'s final and capture his record-breaking 23rd Grand Slam title."\n    },\n    {\n      "title": "Novak Djokovic wins his 23rd Grand Slam title by beating Casper Ruud in ...",\n      "url": "https://apnews.com/article/djokovic-ruud-french-open-roland-garros-final-d7bda9f570b010ea48cf8a05b397291e",\n      "content": "Novak Djokovic wins his 23rd Grand Slam title by beating Casper Ruud in the French Open final\\nSerbia\\u2019s Novak Djokovic celebrates winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023. Serbia\\u2019s Novak Djokovic wears the number 23 on his garment for his 23rd Grand Slam Singles title as he kisses the trophy after winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023. Serbia\\u2019s Novak Djokovic wears the number 23 on his garment for his 23rd Grand Slam Singles title as he kisses the trophy after winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023. Serbia\\u2019s Novak Djokovic wears the number 23 on his garment for his 23rd Grand Slam Singles title as he kisses the trophy after winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023. Serbia\\u2019s Novak Djokovic kisses the trophy as he celebrates winning the men\\u2019s singles final match of the French Open tennis tournament against Norway\\u2019s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris, Sunday, June 11, 2023."\n    },\n    {\n      "title": "Novak Djokovic wins his 23rd Grand Slam title : NPR",\n      "url": "https://www.npr.org/2023/06/11/1181568367/novak-djokovic-tennis-french-open-grand-slam",\n      "content": "Sports\\nNovak Djokovic wins the French Open men\'s singles, securing his 23rd Grand Slam title\\nBy\\nThe Associated Press\\nSerbia\'s Novak Djokovic celebrates winning the men\'s singles final match of the French Open tennis tournament against Norway\'s Casper Ruud in three sets at the Roland Garros stadium in Paris, Sunday.\\n Thibault Camus/AP\\nhide caption\\nSerbia\'s Novak Djokovic celebrates winning the men\'s singles final match of the French Open tennis tournament against Norway\'s Casper Ruud in three sets, 7-6, (7-1), 6-3, 7-5, at the Roland Garros stadium in Paris on Sunday.\\n Thibault Camus/AP\\nhide caption\\nSerbia\'s Novak Djokovic celebrates winning the men\'s singles final match of the French Open tennis tournament against Norway\'s Casper Ruud in three sets at the Roland Garros stadium in Paris, Sunday.\\n At 20 days past his 36th birthday, Djokovic is the oldest singles champion at Roland Garros, considered the most grueling of the majors because of the lengthy, grinding points required by the red clay, which is slower than the grass or hard courts underfoot elsewhere.\\n Djokovic came close to pulling off that feat in 2021, when he won the Australian Open, French Open and Wimbledon and made it all the way to the title match at the U.S. Open before losing to Daniil Medvedev.\\n"\n    },\n    {\n      "title": "Winners of the Men\'s French Open - Topend Sports",\n      "url": "https://www.topendsports.com/events/tennis-grand-slam/french-open/winners-men.htm",\n      "content": "Here are all the winners of the French Tennis Open men\'s title since the first tournament in 1925. The recent tournaments have been dominated by Spanish player Rafael Nadal."\n    },\n    {\n      "title": "Carlos Alcaraz wins the French Open, earning a third Grand Slam title",\n      "url": "https://www.npr.org/2024/06/09/nx-s1-4997726/carlos-alcaraz-wins-french-open-third-grand-slam-title",\n      "content": "PARIS \\u2014 As Carlos Alcaraz began constructing his comeback in Sunday\'s French Open final, a 6-3, 2-6, 5-7, 6-1, 6-2 victory over Alexander Zverev for a first championship at Roland Garros and ..."\n    }\n  ]\n}',
                artifact={
                    'query': 'who won the last french open',
                    'follow_up_questions': None,
                    'answer': 'Novak ...',
                    'images': [
                        'https://www.amny.com/wp-content/uploads/2023/06/AP23162622181176-1200x800.jpg',
                        ...
                        ],
                    'results': [
                        {
                            'title': 'Djokovic ...', 'url': 'https://www.nytimes.com...',
                            'content': "Novak...",
                            'score': 0.99505633,
                            'raw_content': 'Tennis\nNovak ...'
                        },
                        ...
                    ],
                    'response_time': 2.92
                },
                tool_call_id='1',
                name='tavily_search_results_json',
            )

    """  # noqa: E501

    name: str = "tavily_search_results_json"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = TavilyInput
    """The tool response format."""

    version: Literal["v1", "v2"] = "v1"
    """The version of the tool to use. 

    Recommended usage is 'v2', but default is 'v1' for backwards compatibility.

    With 'v2' the tool automatically outputs a string, which can be used as the content
    of a ToolMessage. If ``include_answer`` is False, the string is a json dump of a 
    list of dicts. Each represents a search result and can have the 'content', 'url', 
    and 'title' key-values. If ``include_answer`` is True, the string is a json dump
    of a dictionary that has an "answer" key-value and a "results" key-value. "results"
    contains the list of dicts which is otherwise directly json-ified.

    With 'v1' tool outputs a list of dicts, which must be converted
    to a string before being passed back to the model. The dicts only contain the 'url' 
    and 'content' key-values for each result.
    """

    max_results: int = 5
    """Max search results to return, default is 5"""
    search_depth: str = "advanced"
    '''The depth of the search. It can be "basic" or "advanced"'''
    include_domains: List[str] = []
    """A list of domains to specifically include in the search results. Default is None, which includes all domains."""  # noqa: E501
    exclude_domains: List[str] = []
    """A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains."""  # noqa: E501
    include_answer: bool = False
    """Include a short answer to original query in the search results. Default is False."""  # noqa: E501
    include_raw_content: bool = False
    """Include cleaned and parsed HTML of each site search results. Default is False."""
    include_images: bool = False
    """Include a list of query related images in the response. Default is False."""

    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)  # type: ignore[arg-type]
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool."""
        # TODO: remove try/except, should be handled by BaseTool
        try:
            raw_results = self.api_wrapper.raw_results(
                query,
                self.max_results,
                self.search_depth,
                self.include_domains,
                self.exclude_domains,
                self.include_answer,
                self.include_raw_content,
                self.include_images,
            )
        except Exception as e:
            return repr(e), {}
        return self._format_content(raw_results), raw_results

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query,
                self.max_results,
                self.search_depth,
                self.include_domains,
                self.exclude_domains,
                self.include_answer,
                self.include_raw_content,
                self.include_images,
            )
        except Exception as e:
            return repr(e), {}
        return self._format_content(raw_results), raw_results

    def _format_content(self, raw_results: dict) -> Union[List[Dict[str, str]], str]:
        if self.version == "v2":
            results: Union[list, dict] = [
                {k: res[k] for k in ("title", "url", "content") if k in res}
                for res in raw_results["results"]
            ]
            if self.include_answer:
                results = {"answer": raw_results.get("answer", ""), "results": results}
            return json.dumps(results, indent=2)
        else:
            return self.api_wrapper.clean_results(raw_results["results"])


class TavilyAnswer(BaseTool):
    """Tool that queries the Tavily Search API and gets back an answer."""

    name: str = "tavily_answer"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. "
        "This returns only the answer - not the original source data."
    )
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)  # type: ignore[arg-type]
    args_schema: Type[BaseModel] = TavilyInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.raw_results(
                query,
                max_results=5,
                include_answer=True,
                search_depth="basic",
            )["answer"]
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            result = await self.api_wrapper.raw_results_async(
                query,
                max_results=5,
                include_answer=True,
                search_depth="basic",
            )
            return result["answer"]
        except Exception as e:
            return repr(e)
