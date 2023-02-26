"""Util that calls ArXiv using the API's unofficial python-sdk."""
from typing import Dict, Iterable, Iterator, List

from pydantic.class_validators import root_validator
from pydantic.main import BaseModel


def _get_authors_str(authors: list, first_author: bool = False) -> str:
    """Get a string representation of the authors."""
    output = str()
    if first_author is False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = authors[0]
    return output


class ArXivSearchAPIWrapper(BaseModel):
    """Wrapper around the arXiv Search API

    In order to set this up, you need to install the unofficial python-sdk:
    https://github.com/lukasschwab/arxiv.py

    Example:
        .. code-block:: python
            from langchain import Arxiv
            arxiv = Arxiv()
    """

    max_results: int = 10
    sort_by: str = "relevance"
    sort_order: str = "descending"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that package exists in environment."""
        from arxiv import SortCriterion, SortOrder

        sort_by_mapping = {
            "relevance": SortCriterion.Relevance,
            "lastUpdatedDate": SortCriterion.LastUpdatedDate,
            "submittedDate": SortCriterion.SubmittedDate,
        }
        sort_order_mapping = {
            "ascending": SortOrder.Ascending,
            "descending": SortOrder.Descending,
        }

        values["sort_by"] = sort_by_mapping[values["sort_by"]]
        values["sort_order"] = sort_order_mapping[values["sort_order"]]

        try:
            import arxiv
        except ImportError:
            raise ImportError(
                "arxiv is not installed. " "Please install it with `pip install arxiv`"
            )
        return values

    def run(self, query: str) -> str:
        """Run query through ArXiv Search and parse result."""
        papers = self.results(query)

        # Representation of a list of papers in a string:
        # """
        # Title: a-title
        # Authors: a, b, c
        # Abstract: an-abstract
        #
        #
        # Title: another-title...
        # """
        if len(papers) == 0:
            return "No good Arxiv Search Result was found"

        results = ""
        for paper in papers:
            result = f"Title: {paper['title'].strip()}\n"
            result += f"Authors: {_get_authors_str(paper['authors']).strip()}\n"
            result += f"Abstract: {paper['abstract'].strip()}\n"
            result += "\n\n"
            results += result.strip()

        return results

    def results(self, query: str) -> List[dict]:
        """Return results from ArXiv."""
        return self._parse_results(self._arxiv_search_results(query))

    def _parse_results(self, results: Iterator) -> List[dict]:
        papers = []

        try:
            result = next(results)

            paper_id = result.get_short_id()
            paper_title = result.title
            paper_url = result.entry_id
            paper_abstract = result.summary.replace("\n", " ")
            paper_authors = [str(author) for author in result.authors]
            paper_first_author = _get_authors_str(result.authors, first_author=True)
            primary_category = result.primary_category
            publish_time = result.published.date()
            update_time = result.updated.date()
            comments = result.comment

            paper = {
                "id": paper_id,
                "title": paper_title,
                "url": paper_url,
                "abstract": paper_abstract,
                "authors": paper_authors,
                "first_author": paper_first_author,
                "primary_category": primary_category,
                "publish_time": publish_time,
                "update_time": update_time,
                "comments": comments,
            }
            papers.append(paper)
        except (
            StopIteration
        ):  # See: https://docs.python.org/3/library/stdtypes.html#iterator.__next__
            pass  # No results found

        return papers

    def _arxiv_search_results(self, query: str) -> Iterator:
        """Search arXiv for query and return results."""
        from arxiv import Search

        search_results = Search(
            query=query,
            max_results=self.max_results,
            sort_by=self.sort_by,
            sort_order=self.sort_order,
        )

        return search_results.results()
