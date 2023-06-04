"""Module for initiating a set of searches relevant for answering the question."""
from __future__ import annotations

import asyncio
import typing
from typing import Any, Dict, List, Mapping, Optional, Sequence

from bs4 import BeautifulSoup

from langchain import LLMChain, PromptTemplate, serpapi
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.classification.multiselection import MultiSelectChain
from langchain.schema import BaseOutputParser


def _extract_content_from_tag(html: str, tag: str) -> List[str]:
    """Extract content from the given tag."""
    soup = BeautifulSoup(html, "lxml")
    queries = []
    for query in soup.find_all(tag):
        queries.append(query.text)
    return queries


def _extract_href_tags(html: str) -> List[str]:
    """Extract href tags.

    Args:
        html: the html to extract href tags from.

    Returns:
        a list of href tags.
    """
    href_tags = []
    soup = BeautifulSoup(html, "html.parser")
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href")
        if href:
            href_tags.append(href)
    return href_tags


class QueryExtractor(BaseOutputParser[List[str]]):
    """An output parser that extracts all queries."""

    def parse(self, text: str) -> List[str]:
        """Extract all content of <query> from the text."""
        return _extract_content_from_tag(text, "query")


QUERY_GENERATION_PROMPT = PromptTemplate.from_template(
    """\
Suggest a few different search queries that could be used to identify web-pages \
that could answer the following question.

If the question is about a named entity start by listing very general \
searches (e.g., just the named entity) and then suggest searches more \
scoped to the question.

Input: ```Where did John Snow from Cambridge, UK work?```
Output: <query>John Snow</query>
<query>John Snow Cambridge UK</query>
<query> John Snow Cambridge UK work history </query>
<query> John Snow Cambridge UK cv </query>

Input: ```How many research papers did Jane Doe publish in 2010?```
Output: <query>Jane Doe</query>
<query>Jane Doe research papers</query>
<query>Jane Doe research research</query>
<query>Jane Doe publications</query>
<query>Jane Doe publications 2010</query>

Input: ```What is the capital of France?```
Output: <query>France</query>
<query>France capital</query>
<query>France capital city</query>
<query>France capital city name</query>

Input: ```What are the symptoms of COVID-19?```
Output: <query>COVID-19</query>
<query>COVID-19 symptoms</query>
<query>COVID-19 symptoms list</query>
<query>COVID-19 symptoms list WHO</query>

Input: ```What is the revenue stream of CVS?```
Output: <query>CVS</query>
<query>CVS revenue</query>
<query>CVS revenue stream</query>
<query>CVS revenue stream business model</query>

Input: ```{question}```
Output:
""",
    output_parser=QueryExtractor(),
)


def _deduplicate_objects(
    dicts: Sequence[Mapping[str, Any]], key: str
) -> List[Mapping[str, Any]]:
    """Deduplicate objects by the given key.

    TODO(Eugene): add a way to add weights to the objects.

    Args:
        dicts: a list of dictionaries to deduplicate.
        key: the key to deduplicate by.

    Returns:
        a list of deduplicated dictionaries.
    """
    unique_values = set()
    deduped: List[Mapping[str, Any]] = []

    for d in dicts:
        value = d[key]
        if value not in unique_values:
            unique_values.add(value)
            deduped.append(d)

    return deduped


def _run_searches(queries: Sequence[str], top_k: int = -1) -> List[Mapping[str, Any]]:
    """Run the given queries and return all the search results.

    This function can return duplicated results, de-duplication can take place later
    and take into account the frequency of appearance.

    Args:
        queries: a list of queries to run
        top_k: the number of results to return, if -1 return all results

    Returns:
        a list of unique search results
    """
    wrapper = serpapi.SerpAPIWrapper()
    results = []
    for query in queries:
        result = wrapper.results(query)
        all_organic_results = result.get("organic_results", [])
        if top_k <= 0:
            organic_results = all_organic_results
        else:
            organic_results = all_organic_results[:top_k]
        results.extend(organic_results)
    return results


async def _arun_searches(
    queries: Sequence[str], top_k: int = -1
) -> List[Mapping[str, Any]]:
    """Run the given queries and return all the search results.

    This function can return duplicated results, de-duplication can take place later
    and take into account the frequency of appearance.

    Args:
        queries: a list of queries to run
        top_k: the number of results to return, if -1 return all results

    Returns:
        a list of unique search results
    """
    wrapper = serpapi.SerpAPIWrapper()
    tasks = [wrapper.aresults(query) for query in queries]
    results = await asyncio.gather(*tasks)

    finalized_results = []

    for result in results:
        all_organic_results = result.get("organic_results", [])
        if top_k <= 0:
            organic_results = all_organic_results
        else:
            organic_results = all_organic_results[:top_k]

        finalized_results.extend(organic_results)

    return finalized_results


# PUBLIC API


def make_query_generator(llm: BaseLanguageModel) -> LLMChain:
    """Use an LLM to break down a complex query into a list of simpler queries.

    The simpler queries are used to run searches against a search engine in a goal
    to increase the recall step and retrieve as many relevant results as possible.


    Query:

    Does Harrison Chase of Langchain like to eat pizza or play squash?

    May be broken down into a list of simpler queries like:

    - Harrison Chase
    - Harrison Chase Langchain
    - Harrison Chase Langchain pizza
    - Harrison Chase Langchain squash
    """
    return LLMChain(
        llm=llm,
        output_key="urls",
        prompt=QUERY_GENERATION_PROMPT,
    )


class GenericSearcher(Chain):
    """A chain that takes a complex question and identifies a list of relevant urls.

    The chain works by:

    1. Breaking a complex question into a series of simpler queries using an LLM.
    2. Running the queries against a search engine.
    3. Selecting the most relevant urls using an LLM (can be replaced with tf-idf
       or other models).

    This chain is not meant to be used for questions requiring multiple hops to answer.

    For example, the age of leonardo dicaprio's girlfriend is a multi-hop question

    This kind of question requires a slightly different approach.

    This chain is meant to handle questions for which one wants
    to collect information from multiple sources.

    To extend implementation:
    * Parameterize the search engine (to allow for non serp api search engines)
    * Expose an abstract interface for query generator and link selection model
    * Expose promoted answers from search engines as blobs that should be summarized
    """

    query_generator: LLMChain
    """An LLM used to break down a complex question into a list of simpler queries."""
    link_selection_model: Chain
    """An LLM that is used to select the most relevant urls from the search results."""
    top_k_per_search: int = -1
    """The number of top urls to select from each search."""

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["urls"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        question = inputs["question"]
        queries = typing.cast(
            List[str],
            self.query_generator.predict_and_parse(
                callbacks=run_manager.get_child() if run_manager else None,
                question=question,
            ),
        )
        results = _run_searches(queries, top_k=self.top_k_per_search)
        deuped_results = _deduplicate_objects(results, "link")
        records = [
            {"link": result["link"], "title": result["title"]}
            for result in deuped_results
        ]
        response_ = self.link_selection_model(
            {
                "question": question,
                "choices": records,
            },
            callbacks=run_manager.get_child() if run_manager else None,
        )
        return {"urls": [result["link"] for result in response_["selected"]]}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        question = inputs["question"]
        queries = typing.cast(
            List[str],
            await self.query_generator.apredict_and_parse(
                callbacks=run_manager.get_child() if run_manager else None,
                question=question,
            ),
        )
        results = _run_searches(queries, top_k=self.top_k_per_search)
        deuped_results = _deduplicate_objects(results, "link")
        records = [
            {"link": result["link"], "title": result["title"]}
            for result in deuped_results
        ]
        response_ = await self.link_selection_model.acall(
            {
                "question": question,
                "choices": records,
            },
            callbacks=run_manager.get_child() if run_manager else None,
        )
        return {"urls": [result["link"] for result in response_["selected"]]}

    @classmethod
    def from_llms(
        cls,
        link_selection_llm: BaseLanguageModel,
        query_generation_llm: BaseLanguageModel,
        *,
        top_k_per_search: int = -1,
    ) -> GenericSearcher:
        """Initialize the searcher from a language model."""
        link_selection_model = MultiSelectChain.from_default(llm=link_selection_llm)
        query_generation_model = make_query_generator(query_generation_llm)
        return cls(
            link_selection_model=link_selection_model,
            query_generator=query_generation_model,
            top_k_per_search=top_k_per_search,
        )
