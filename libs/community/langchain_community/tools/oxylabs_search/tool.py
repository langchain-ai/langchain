"""Tool for the Oxylabs Search API."""

import json
from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_community.utilities.oxylabs_search import OxylabsSearchAPIWrapper


class OxylabsSearchQueryInput(BaseModel):
    """Input for the OxylabsSearch tool."""

    query: str = Field(description="query to retrieve on Oxylabs Search API")
    geo_location: Optional[str] = Field(
        default="California,United States",
        description="Geographic location for the search;"
        " adjust if location-specific information is requested.",
    )


class OxylabsSearchRun(BaseTool):  # type: ignore[override, override]
    """Oxylabs Search Run tool.

    Setup:
        Install ``langchain-community``, ``oxylabs``,
        and set environment variables ``OXYLABS_USERNAME`` and ``OXYLABS_PASSWORD``.

        .. code-block:: bash

            pip install langchain-community, oxylabs
            export OXYLABS_USERNAME="your-oxylabs-username"
            export OXYLABS_PASSWORD="your-oxylabs-password"

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.oxylabs_search import OxylabsSearchRun
            from langchain_community.utilities import OxylabsSearchAPIWrapper

            oxylabs_wrapper = OxylabsSearchAPIWrapper()
            tool = OxylabsSearchRun(wrapper=oxylabs_wrapper)

    Invocation with args:
        .. code-block:: python

            tool_.invoke({"query": "Visit restaurants in Vilnius."})

    Invocation with ToolCall:

        .. code-block:: python

            tool = OxylabsSearchRun(wrapper=oxylabs_wrapper, kwargs={"result_categories": ["local_information", "combined_search_result"]})
            model_generated_tool_call = {"args": {"query": "Visit restaurants in Vilnius.", "geo_location": "Vilnius,Lithuania"}, "id": "1", "name": "oxylabs_search", "type": "tool_call",}
            tool.invoke(model_generated_tool_call)

        .. code-block:: python

            ToolMessage(
                ('  LOCAL PACK: \n'
                 '  ITEMS ITEMS: \n'
                 '    ITEMS-ITEM-1: \n'
                 '    CID: 13950149882539119249\n'
                 '    POS: 1\n'
                 '    TITLE: Etno Dvaras\n'
                 '    RATING: 4.5\n'
                 '    ADDRESS: Lithuanian\n'
                 '    ITEMS-ITEM-2: \n'
                 '    CID: 711702509070991018\n'
                 '    POS: 2\n'
                 '    TITLE: Lokys\n'
                 '    RATING: 4.5\n'
                 '    ADDRESS: Lithuanian\n'
                 '    ITEMS-ITEM-3: \n'
                 '    CID: 7630589473191639738\n'
                 '    POS: 3\n'
                 '    TITLE: Senoji trobelė\n'
                 '    RATING: 4.4\n'
                 '    ADDRESS: Lithuanian\n'
                 '\n'
                 '  ORGANIC RESULTS ITEMS: \n'
                 '    ORGANIC-ITEM-1: \n'
                 '    POS: 1\n'
                 '    URL: '
                 'https://www.tripadvisor.com/Restaurants-g274951-Vilnius_Vilnius_County.html\n'
                 '    DESC: Some of the best restaurants in Vilnius for families with children '
                 'include: Momo grill Vilnius · Jurgis ir Drakonas Ogmios · RoseHip Vegan\xa0'
                 '...\n'
                 '    TITLE: THE 10 BEST Restaurants in Vilnius (Updated November ...\n'
                 '    SITELINKS: \n'
                 '      SITELINKS: \n'
                 '      INLINE ITEMS: \n'
                 '        INLINE-ITEM-1: \n'
                 '        URL: '
                 'https://www.tripadvisor.com/Restaurants-g274951-zfp58-Vilnius_Vilnius_County.html\n'
                 '        TITLE: Vilnius Dinner Restaurants\n'
                 '        INLINE-ITEM-2: \n'
                 '        URL: '
                 'https://www.tripadvisor.com/Restaurants-g274951-zfp2-Vilnius_Vilnius_County.html\n'
                 '        TITLE: Vilnius Breakfast Restaurants\n'
                 '        INLINE-ITEM-3: \n'
                 '        URL: '
                 'https://www.tripadvisor.com/Restaurants-g274951-c8-Vilnius_Vilnius_County.html\n'
                 '        TITLE: Cafés in Vilnius\n'
                 '    URL_SHOWN: https://www.tripadvisor.com› ... › Vilnius\n'
                 '    FAVICON_TEXT: Tripadvisor\n'
                 '    ORGANIC-ITEM-2: \n'
                 '    POS: 2\n'
                 '    URL: '
                 'https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania\n'
                 "    DESC: Jun 24, 2024 — Lithuania's capital has established itself as an "
                 "affordable culinary hotspot as four of the city's restaurants awarded "
                 'Michelin stars.\n'
                 '    TITLE: Star quality: a foodie guide to Vilnius\n'
                 '    URL_SHOWN: https://theweek.com› Culture & Life › Food & Drink\n'
                 '    FAVICON_TEXT: The Week\n'
                 '    ...
            )

    """  # noqa: E501

    name: str = "oxylabs_search"
    description: str = (
        "A meta search engine."
        "Ideal for situations where you need to answer questions about current events,"
        "facts, products, recipes, local information, and other topics "
        "that can be explored via web browsing. "
        "The input should be a search query and, if applicable,"
        " a geo_location string to enhance result accuracy. "
        "The output is a compiled, formatted summary of query results. "
    )
    wrapper: OxylabsSearchAPIWrapper
    kwargs: dict = Field(default_factory=dict)
    args_schema: Type[BaseModel] = OxylabsSearchQueryInput

    def _run(
        self,
        query: str,
        geo_location: Optional[str] = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        kwargs_ = dict(**self.kwargs)
        kwargs_.update(
            {
                "geo_location": geo_location,
            }
        )

        return self.wrapper.run(query, **kwargs_)

    async def _arun(
        self,
        query: str,
        geo_location: Optional[str] = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""

        kwargs_ = dict(**self.kwargs)
        kwargs_.update(
            {
                "geo_location": geo_location,
            }
        )

        return await self.wrapper.arun(query, **kwargs_)


class OxylabsSearchResults(BaseTool):  # type: ignore[override, override]
    """Oxylabs Search Results tool.

    Setup:
        Install ``langchain-community``, ``oxylabs``,
        and set environment variables ``OXYLABS_USERNAME`` and ``OXYLABS_PASSWORD``.

        .. code-block:: bash

            pip install langchain-community, oxylabs
            export OXYLABS_USERNAME="your-oxylabs-username"
            export OXYLABS_PASSWORD="your-oxylabs-password"

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.oxylabs_search import OxylabsSearchRun
            from langchain_community.utilities import OxylabsSearchAPIWrapper

            oxylabs_wrapper = OxylabsSearchAPIWrapper()
            tool = OxylabsSearchResults(wrapper=oxylabs_wrapper)

    Invocation with args:
        .. code-block:: python

            response_results = tool.invoke({"query": "Visit restaurants in Vilnius."})
            response_results = json.loads(response_results)
            for item in response_results:
                print(item)

    Invocation with ToolCall:

        .. code-block:: python

            model_generated_tool_call = {"args": {"query": "Visit restaurants in Vilnius.", "geo_location": "Vilnius,Lithuania"}, "id": "1", "name": "oxylabs_search", "type": "tool_call",}
            tool.invoke(model_generated_tool_call)

        .. code-block:: python

            ToolMessage('[{"paid": [], "organic": [{"pos": 1, "url": '
             '"https://www.tripadvisor.com/Restaurants-g274951-Vilnius_Vilnius_County.html", '
             '"desc": "Some of the best restaurants in Vilnius for families with children '
             'include: Momo grill Vilnius \\u00b7 Jurgis ir Drakonas Ogmios \\u00b7 '
             'RoseHip Vegan\\u00a0...", "title": "THE 10 BEST Restaurants in Vilnius '
             '(Updated November ...", "sitelinks": {"inline": [{"url": '
             '"https://www.tripadvisor.com/Restaurants-g274951-zfp58-Vilnius_Vilnius_County.html", '
             '"title": "Vilnius Dinner Restaurants"}, {"url": '
             '"https://www.tripadvisor.com/Restaurants-g274951-zfp2-Vilnius_Vilnius_County.html", '
             '"title": "Vilnius Breakfast Restaurants"}, {"url": '
             '"https://www.tripadvisor.com/Restaurants-g274951-c8-Vilnius_Vilnius_County.html", '
             '"title": "Caf\\u00e9s in Vilnius"}]}, "url_shown": '
             '"https://www.tripadvisor.com\\u203a ... \\u203a Vilnius", "pos_overall": 1, '
             '"favicon_text": "Tripadvisor"}, {"pos": 2, "url": '
             '"https://www.amsterdamfoodie.nl/2022/foodie-guide-to-vilnius-lithuania/", '
             '"desc": "Dec 2, 2022 \\u2014 Vilnius takes the top spot for most creative '
             'dining with the fabulous restaurant Amandus. Modern gastronomy, with a nod '
             'to Lithuanian tradition, goes hand in\\u00a0...", "title": "A Foodie\'s '
             'Guide to Vilnius, Lithuania", "url_shown": '
             '"https://www.amsterdamfoodie.nl\\u203a blog", "pos_overall": 2, '
             '"favicon_text": "Amsterdam Foodie"}, {"pos": 3, "url": '
             '"https://www.reddit.com/r/Vilnius/comments/uudd1p/good_restaurants_street_food_in_vilnius/", '
             '"desc": "- Alin\\u0117 Lei\\u010diai - also traditional peasant food of '
             'history but in a more upscale environment, and more expensive, table service '
             'and more diverse\\u00a0...", "title": "Good restaurants & street food in '
             'Vilnius?", "url_shown": "10+ comments  \\u00b7  2 years ago", "pos_overall": '
             '3, "favicon_text": "Reddit\\u00a0\\u00b7\\u00a0r/Vilnius"}, {"pos": 4, '
             '"url": '
             '"https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania", '
             '"desc": "Jun 24, 2024 \\u2014 Lithuania\'s capital has established itself as '
             "an affordable culinary hotspot as four of the city's restaurants awarded "
             'Michelin stars.", "title": "Star quality: a foodie guide to Vilnius", '
             '"sitelinks": {"inline": [{"url": '
             '"https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania#:~:text=Nineteen18,-With%20its%20%60%60sleek%20and", '
             '"title": "Nineteen18"}, {"url": '
             '"https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania#:~:text=D%C5%BEiaugsmas,-D%C5%BEiaugsmas", '
             '"title": "D\\u017eiaugsmas"}, {"url": '
             '"https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania#:~:text=Gaspar%27s", '
             '"title": "Gaspar\'s"}]}, "url_shown": "https://theweek.com\\u203a Culture & '
             'Life \\u203a Food & Drink", "pos_overall": 4, "favicon_text": "The Week"}], '
             '"local_pack": {"items": [{"cid": "13950149882539119249", "pos": 1, "title": '
             '"Etno Dvaras", "rating": 4.5, "address": "Lithuanian", "rating_count": 0}, '
             '{"cid": "7630589473191639738", "pos": 2, "title": "Senoji trobel\\u0117", '
             '"rating": 4.4, "address": "Lithuanian", "rating_count": 0}, {"cid": '
             '"6362402554387935672", "pos": 3, "title": "Ertlio Namas", "rating": 4.7, '
             '"address": "\\u0160v. Jono g. 7", "subtitle": "Lithuanian", "rating_count": '
             '0}], "pos_overall": 5}, "related_searches": {"pos_overall": 6, '
             '"related_searches": ["Best restaurants in Vilnius old town", "Unique places '
             'to eat in Vilnius", "Best visit restaurants in vilnius", "Best Lithuanian '
             'restaurants in Vilnius", "Restaurants Vilnius old Town", "Vilnius '
             'restaurants Michelin", "Fine dining Vilnius", "Tripadvisor Vilnius '
             'Restaurants"]}, "search_information": {"query": "Visit restaurants in '
             'Vilnius.", "geo_location": "United States", "showing_results_for": "Visit '
             'restaurants in Vilnius.", "total_results_count": 1260000}, '
             '"total_results_count": 1260000}]'
         )
    """  # noqa: E501

    name: str = "oxylabs_search_results"
    description: str = (
        "A meta search engine."
        "Ideal for situations where you need to answer questions about current events,"
        "facts, products, recipes, local information, and other topics "
        "that can be explored via web browsing. "
        "The input should be a search query and, if applicable,"
        " a geo_location string to enhance result accuracy. "
        "The output is a JSON array of response page objects. "
    )
    wrapper: OxylabsSearchAPIWrapper
    kwargs: dict = Field(default_factory=dict)
    args_schema: Type[BaseModel] = OxylabsSearchQueryInput

    model_config = ConfigDict(
        extra="allow",
    )

    def _run(
        self,
        query: str,
        geo_location: Optional[str] = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        kwargs_ = dict(**self.kwargs)
        kwargs_.update(
            {
                "geo_location": geo_location,
            }
        )

        return json.dumps(self.wrapper.results(query, **kwargs_))

    async def _arun(
        self,
        query: str,
        geo_location: Optional[str] = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""

        kwargs_ = dict(**self.kwargs)
        kwargs_.update(
            {
                "geo_location": geo_location,
            }
        )

        return json.dumps(await self.wrapper.aresults(query, **kwargs_))
