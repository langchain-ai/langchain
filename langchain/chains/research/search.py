"""Module for initiating a set of searches relevant for answering the question."""
import asyncio
import itertools
from bs4 import BeautifulSoup
from typing import Sequence, List, Mapping, Any

from langchain import PromptTemplate, LLMChain, serpapi
from langchain.base_language import BaseLanguageModel
from langchain.chains.research.typedefs import AbstractQueryGenerator, AbstractSearcher
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


# TODO(Eugene): add a version that works for chat models as well w/ human system message?
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


def _run_searches(queries: Sequence[str]) -> List[Mapping[str, Any]]:
    """Run the given queries and return the unique results.

    Args:
        queries: a list of queries to run

    Returns:
        a list of unique search results
    """
    wrapper = serpapi.SerpAPIWrapper()
    results = []
    for query in queries:
        result = wrapper.results(query)
        organic_results = result["organic_results"]
        results.extend(organic_results)
    return results


async def _arun_searches(queries: Sequence[str]) -> List[Mapping[str, Any]]:
    """Run the given queries and return the unique results."""
    wrapper = serpapi.SerpAPIWrapper()
    tasks = [wrapper.results(query) for query in queries]
    results = await asyncio.gather(*tasks)

    return list(
        itertools.chain.from_iterable(result["organic_results"] for result in results)
    )


# PUBLIC API


class GenericQueryGenerator(AbstractQueryGenerator):
    def __init__(self, llm: BaseLanguageModel) -> None:
        """Initialize the query generator."""
        self.llm = llm

    def generate_queries(self, question: str) -> List[str]:
        """Generate queries using a Chain."""
        chain = LLMChain(prompt=QUERY_GENERATION_PROMPT, llm=self.llm)
        queries = chain.predict_and_parse(question=question)
        return queries


class GenericSearcher(AbstractSearcher):
    def __init__(self, search_engine: str = "serp") -> None:
        """Initialize the searcher.

        Args:
            search_engine: the search engine to use. Placeholder for future work.
        """
        if search_engine != "serp":
            raise NotImplementedError("Only serp is supported at the moment.")

    def asearch(self, queries: Sequence[str]) -> List[Mapping[str, Any]]:
        """Run a search for the given query."""
        results = await _arun_searches(queries)
        return _deduplicate_objects(results, "link")
