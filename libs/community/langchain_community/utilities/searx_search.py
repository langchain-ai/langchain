"""Utility for using SearxNG meta search API.

SearxNG is a privacy-friendly free metasearch engine that aggregates results from
`multiple search engines
<https://docs.searxng.org/admin/engines/configured_engines.html>`_ and databases and
supports the `OpenSearch
<https://github.com/dewitt/opensearch/blob/master/opensearch-1-1-draft-6.md>`_
specification.

More details on the installation instructions `here. <../../integrations/searx.html>`_

For the search API refer to https://docs.searxng.org/dev/search_api.html

Quick Start
-----------


In order to use this utility you need to provide the searx host. This can be done
by passing the named parameter :attr:`searx_host <SearxSearchWrapper.searx_host>`
or exporting the environment variable SEARX_HOST.
Note: this is the only required parameter.

Then create a searx search instance like this:

    .. code-block:: python

        from langchain_community.utilities import SearxSearchWrapper

        # when the host starts with `http` SSL is disabled and the connection
        # is assumed to be on a private network
        searx_host='http://self.hosted'

        search = SearxSearchWrapper(searx_host=searx_host)


You can now use the ``search`` instance to query the searx API.

Searching
---------

Use the :meth:`run() <SearxSearchWrapper.run>` and
:meth:`results() <SearxSearchWrapper.results>` methods to query the searx API.
Other methods are available for convenience.

:class:`SearxResults` is a convenience wrapper around the raw json result.

Example usage of the ``run`` method to make a search:

    .. code-block:: python

        s.run(query="what is the best search engine?")

Engine Parameters
-----------------

You can pass any `accepted searx search API
<https://docs.searxng.org/dev/search_api.html>`_ parameters to the
:py:class:`SearxSearchWrapper` instance.

In the following example we are using the
:attr:`engines <SearxSearchWrapper.engines>` and the ``language`` parameters:

    .. code-block:: python

        # assuming the searx host is set as above or exported as an env variable
        s = SearxSearchWrapper(engines=['google', 'bing'],
                            language='es')

Search Tips
-----------

Searx offers a special
`search syntax <https://docs.searxng.org/user/index.html#search-syntax>`_
that can also be used instead of passing engine parameters.

For example the following query:

    .. code-block:: python

        s = SearxSearchWrapper("langchain library", engines=['github'])

        # can also be written as:
        s = SearxSearchWrapper("langchain library !github")
        # or even:
        s = SearxSearchWrapper("langchain library !gh")


In some situations you might want to pass an extra string to the search query.
For example when the `run()` method is called by an agent. The search suffix can
also be used as a way to pass extra parameters to searx or the underlying search
engines.

    .. code-block:: python

        # select the github engine and pass the search suffix
        s = SearchWrapper("langchain library", query_suffix="!gh")


        s = SearchWrapper("langchain library")
        # select github the conventional google search syntax
        s.run("large language models", query_suffix="site:github.com")


*NOTE*: A search suffix can be defined on both the instance and the method level.
The resulting query will be the concatenation of the two with the former taking
precedence.


See `SearxNG Configured Engines
<https://docs.searxng.org/admin/engines/configured_engines.html>`_ and
`SearxNG Search Syntax <https://docs.searxng.org/user/index.html#id1>`_
for more details.

Notes
-----
This wrapper is based on the SearxNG fork https://github.com/searxng/searxng which is
better maintained than the original Searx project and offers more features.

Public searxNG instances often use a rate limiter for API usage, so you might want to
use a self hosted instance and disable the rate limiter.

If you are self-hosting an instance you can customize the rate limiter for your
own network as described
`here <https://docs.searxng.org/src/searx.botdetection.html#limiter-src>`_.


For a list of public SearxNG instances see https://searx.space/
"""

import json
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    PrivateAttr,
    root_validator,
    validator,
)
from langchain_core.utils import get_from_dict_or_env


def _get_default_params() -> dict:
    return {"language": "en", "format": "json"}


class SearxResults(dict):
    """Dict like wrapper around search api results."""

    _data: str = ""

    def __init__(self, data: str):
        """Take a raw result from Searx and make it into a dict like object."""
        json_data = json.loads(data)
        super().__init__(json_data)
        self.__dict__ = self

    def __str__(self) -> str:
        """Text representation of searx result."""
        return self._data

    @property
    def results(self) -> Any:
        """Silence mypy for accessing this field.

        :meta private:
        """
        return self.get("results")

    @property
    def answers(self) -> Any:
        """Helper accessor on the json result."""
        return self.get("answers")


class SearxSearchWrapper(BaseModel):
    """Wrapper for Searx API.

    To use you need to provide the searx host by passing the named parameter
    ``searx_host`` or exporting the environment variable ``SEARX_HOST``.

    In some situations you might want to disable SSL verification, for example
    if you are running searx locally. You can do this by passing the named parameter
    ``unsecure``. You can also pass the host url scheme as ``http`` to disable SSL.

    Example:
        .. code-block:: python

            from langchain_community.utilities import SearxSearchWrapper
            searx = SearxSearchWrapper(searx_host="http://localhost:8888")

    Example with SSL disabled:
        .. code-block:: python

            from langchain_community.utilities import SearxSearchWrapper
            # note the unsecure parameter is not needed if you pass the url scheme as
            # http
            searx = SearxSearchWrapper(searx_host="http://localhost:8888",
                                                    unsecure=True)


    """

    _result: SearxResults = PrivateAttr()
    searx_host: str = ""
    unsecure: bool = False
    params: dict = Field(default_factory=_get_default_params)
    headers: Optional[dict] = None
    engines: Optional[List[str]] = []
    categories: Optional[List[str]] = []
    query_suffix: Optional[str] = ""
    k: int = 10
    aiosession: Optional[Any] = None

    @validator("unsecure")
    def disable_ssl_warnings(cls, v: bool) -> bool:
        """Disable SSL warnings."""
        if v:
            # requests.urllib3.disable_warnings()
            try:
                import urllib3

                urllib3.disable_warnings()
            except ImportError as e:
                print(e)  # noqa: T201

        return v

    @root_validator(pre=True)
    def validate_params(cls, values: Dict) -> Dict:
        """Validate that custom searx params are merged with default ones."""
        user_params = values.get("params", {})
        default = _get_default_params()
        values["params"] = {**default, **user_params}

        engines = values.get("engines")
        if engines:
            values["params"]["engines"] = ",".join(engines)

        categories = values.get("categories")
        if categories:
            values["params"]["categories"] = ",".join(categories)

        searx_host = get_from_dict_or_env(values, "searx_host", "SEARX_HOST")
        if not searx_host.startswith("http"):
            print(  # noqa: T201
                f"Warning: missing the url scheme on host \
                ! assuming secure https://{searx_host} "
            )
            searx_host = "https://" + searx_host
        elif searx_host.startswith("http://"):
            values["unsecure"] = True
            cls.disable_ssl_warnings(True)
        values["searx_host"] = searx_host

        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _searx_api_query(self, params: dict) -> SearxResults:
        """Actual request to searx API."""
        raw_result = requests.get(
            self.searx_host,
            headers=self.headers,
            params=params,
            verify=not self.unsecure,
        )
        # test if http result is ok
        if not raw_result.ok:
            raise ValueError("Searx API returned an error: ", raw_result.text)
        res = SearxResults(raw_result.text)
        self._result = res
        return res

    async def _asearx_api_query(self, params: dict) -> SearxResults:
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.searx_host,
                    headers=self.headers,
                    params=params,
                    ssl=(lambda: False if self.unsecure else None)(),
                ) as response:
                    if not response.ok:
                        raise ValueError("Searx API returned an error: ", response.text)
                    result = SearxResults(await response.text())
                    self._result = result
        else:
            async with self.aiosession.get(
                self.searx_host,
                headers=self.headers,
                params=params,
                verify=not self.unsecure,
            ) as response:
                if not response.ok:
                    raise ValueError("Searx API returned an error: ", response.text)
                result = SearxResults(await response.text())
                self._result = result

        return result

    def run(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> str:
        """Run query through Searx API and parse results.

        You can pass any other params to the searx query API.

        Args:
            query: The query to search for.
            query_suffix: Extra suffix appended to the query.
            engines: List of engines to use for the query.
            categories: List of categories to use for the query.
            **kwargs: extra parameters to pass to the searx API.

        Returns:
            str: The result of the query.

        Raises:
            ValueError: If an error occurred with the query.


        Example:
            This will make a query to the qwant engine:

            .. code-block:: python

                from langchain_community.utilities import SearxSearchWrapper
                searx = SearxSearchWrapper(searx_host="http://my.searx.host")
                searx.run("what is the weather in France ?", engine="qwant")

                # the same result can be achieved using the `!` syntax of searx
                # to select the engine using `query_suffix`
                searx.run("what is the weather in France ?", query_suffix="!qwant")
        """
        _params = {
            "q": query,
        }
        params = {**self.params, **_params, **kwargs}

        if self.query_suffix and len(self.query_suffix) > 0:
            params["q"] += " " + self.query_suffix

        if isinstance(query_suffix, str) and len(query_suffix) > 0:
            params["q"] += " " + query_suffix

        if isinstance(engines, list) and len(engines) > 0:
            params["engines"] = ",".join(engines)

        if isinstance(categories, list) and len(categories) > 0:
            params["categories"] = ",".join(categories)

        res = self._searx_api_query(params)

        if len(res.answers) > 0:
            toret = res.answers[0]

        # only return the content of the results list
        elif len(res.results) > 0:
            toret = "\n\n".join([r.get("content", "") for r in res.results[: self.k]])
        else:
            toret = "No good search result found"

        return toret

    async def arun(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> str:
        """Asynchronously version of `run`."""
        _params = {
            "q": query,
        }
        params = {**self.params, **_params, **kwargs}

        if self.query_suffix and len(self.query_suffix) > 0:
            params["q"] += " " + self.query_suffix

        if isinstance(query_suffix, str) and len(query_suffix) > 0:
            params["q"] += " " + query_suffix

        if isinstance(engines, list) and len(engines) > 0:
            params["engines"] = ",".join(engines)

        res = await self._asearx_api_query(params)

        if len(res.answers) > 0:
            toret = res.answers[0]

        # only return the content of the results list
        elif len(res.results) > 0:
            toret = "\n\n".join([r.get("content", "") for r in res.results[: self.k]])
        else:
            toret = "No good search result found"

        return toret

    def results(
        self,
        query: str,
        num_results: int,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> List[Dict]:
        """Run query through Searx API and returns the results with metadata.

        Args:
            query: The query to search for.
            query_suffix: Extra suffix appended to the query.
            num_results: Limit the number of results to return.
            engines: List of engines to use for the query.
            categories: List of categories to use for the query.
            **kwargs: extra parameters to pass to the searx API.

        Returns:
            Dict with the following keys:
            {
                snippet:  The description of the result.
                title:  The title of the result.
                link: The link to the result.
                engines: The engines used for the result.
                category: Searx category of the result.
            }

        """
        _params = {
            "q": query,
        }
        params = {**self.params, **_params, **kwargs}
        if self.query_suffix and len(self.query_suffix) > 0:
            params["q"] += " " + self.query_suffix
        if isinstance(query_suffix, str) and len(query_suffix) > 0:
            params["q"] += " " + query_suffix
        if isinstance(engines, list) and len(engines) > 0:
            params["engines"] = ",".join(engines)
        if isinstance(categories, list) and len(categories) > 0:
            params["categories"] = ",".join(categories)
        results = self._searx_api_query(params).results[:num_results]
        if len(results) == 0:
            return [{"Result": "No good Search Result was found"}]

        return [
            {
                "snippet": result.get("content", ""),
                "title": result["title"],
                "link": result["url"],
                "engines": result["engines"],
                "category": result["category"],
            }
            for result in results
        ]

    async def aresults(
        self,
        query: str,
        num_results: int,
        engines: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> List[Dict]:
        """Asynchronously query with json results.

        Uses aiohttp. See `results` for more info.
        """
        _params = {
            "q": query,
        }
        params = {**self.params, **_params, **kwargs}

        if self.query_suffix and len(self.query_suffix) > 0:
            params["q"] += " " + self.query_suffix
        if isinstance(query_suffix, str) and len(query_suffix) > 0:
            params["q"] += " " + query_suffix
        if isinstance(engines, list) and len(engines) > 0:
            params["engines"] = ",".join(engines)
        results = (await self._asearx_api_query(params)).results[:num_results]
        if len(results) == 0:
            return [{"Result": "No good Search Result was found"}]

        return [
            {
                "snippet": result.get("content", ""),
                "title": result["title"],
                "link": result["url"],
                "engines": result["engines"],
                "category": result["category"],
            }
            for result in results
        ]
