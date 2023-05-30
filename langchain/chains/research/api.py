import abc
import asyncio
import json
import urllib.parse
from bs4 import BeautifulSoup, PageElement
from typing import Sequence, List, Mapping, Any, Dict, Tuple, Optional

from langchain import PromptTemplate, LLMChain, serpapi
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
    AsyncCallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.classification.multiselection import (
    IDParser,
    _extract_content_from_tag,
    MultiSelectChain,
)
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.html.markdownify import MarkdownifyHTMLParser
from langchain.schema import BaseOutputParser, Document
from langchain.text_splitter import TextSplitter

Parser = MarkdownifyHTMLParser(tags_to_remove=("svg", "img", "script", "style", "a"))


class AHrefExtractor(BaseOutputParser[List[str]]):
    """An output parser that extracts all a-href links."""

    def parse(self, text: str) -> List[str]:
        return _extract_href_tags(text)


URL_CRAWLING_PROMPT = PromptTemplate.from_template(
    """\
Here is a list of URLs extracted from a page titled: `{title}`.

```csv
{urls}
```

---

Here is a question:

{question}

---


Please output the ids of the URLs that may contain content relevant to answer the question. \
Use only the information csv table of URLs to determine relevancy.

Format your answer inside of an <ids> tags, separating the ids by a comma.

For example, if the 132 and 133 URLs are relevant, you would write: <ids>132,133</ids>

Begin:""",
    output_parser=IDParser(),
)


def _get_surrounding_text(tag: PageElement, n: int, *, is_before: bool = True) -> str:
    """Get surrounding text the given tag in the given direction.

    Args:
        tag: the tag to get surrounding text for.
        n: number of characters to get
        is_before: Whether to get text before or after the tag.

    Returns:
        the surrounding text in the given direction.
    """
    text = ""
    current = tag.previous_element if is_before else tag.next_element

    while current and len(text) < n:
        current_text = str(current.text).strip()
        current_text = (
            current_text
            if len(current_text) + len(text) <= n
            else current_text[: n - len(text)]
        )

        if is_before:
            text = current_text + " " + text
        else:
            text = text + " " + current_text

        current = current.previous_element if is_before else current.next_element

    return text


def get_ahref_snippets(html: str, num_chars: int = 0) -> Dict[str, Any]:
    """Get a list of <a> tags as snippets from the given html.

    Args:
        html: the html to get snippets from.
        num_chars: the number of characters to get around the <a> tags.

    Returns:
        a list of snippets.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip()
    snippets = []

    for idx, a_tag in enumerate(soup.find_all("a")):
        before_text = _get_surrounding_text(a_tag, num_chars, is_before=True)
        after_text = _get_surrounding_text(a_tag, num_chars, is_before=False)
        snippet = {
            "id": idx,
            "before": before_text.strip().replace("\n", " "),
            "link": a_tag.get("href").replace("\n", " ").strip(),
            "content": a_tag.text.replace("\n", " ").strip(),
            "after": after_text.strip().replace("\n", " "),
        }
        snippets.append(snippet)

    return {
        "snippets": snippets,
        "title": title,
    }


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
Suggest a few different search queries that could be used to identify web-pages that could answer \
the following question.

If the question is about a named entity start by listing very general searches (e.g., just the named entity) \
and then suggest searches more scoped to the question.

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


def generate_queries(llm: BaseLanguageModel, question: str) -> List[str]:
    """Generate queries using a Chain."""
    chain = LLMChain(prompt=QUERY_GENERATION_PROMPT, llm=llm)
    queries = chain.predict_and_parse(question=question)
    return queries


def _deduplicate_objects(
    dicts: Sequence[Mapping[str, Any]], key: str
) -> List[Mapping[str, Any]]:
    """Deduplicate objects by the given key."""
    unique_values = set()
    deduped: List[Mapping[str, Any]] = []

    for d in dicts:
        value = d[key]
        if value not in unique_values:
            unique_values.add(value)
            deduped.append(d)

    return deduped


def run_searches(queries: Sequence[str]) -> List[Mapping[str, Any]]:
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

    unique_results = _deduplicate_objects(results, "link")
    return unique_results


class BlobCrawler(abc.ABC):
    """Crawl a blob and identify links to related content."""

    @abc.abstractmethod
    def crawl(self, blob: Blob, query: str) -> List[str]:
        """Explore the blob and identify links to related content that is relevant to the query."""


def _extract_records(blob: Blob) -> Tuple[List[Mapping[str, Any]], Tuple[str, ...]]:
    """Extract records from a blob."""
    if blob.mimetype == "text/html":
        info = get_ahref_snippets(blob.as_string(), num_chars=100)
        return (
            [
                {
                    "content": d["content"],
                    "link": d["link"],
                    "before": d["before"],
                    "after": d["after"],
                }
                for d in info["snippets"]
            ],
            ("link", "content", "before", "after"),
        )
    elif blob.mimetype == "application/json":  # Represent search results
        data = json.loads(blob.as_string())
        results = data["results"]
        return [
            {
                "title": result["title"],
                "snippet": result["snippet"],
                "link": result["link"],
            }
            for result in results
        ], ("link", "title", "snippet")
    else:
        raise ValueError(
            "Can only extract records from HTML/JSON blobs. Got {blob.mimetype}"
        )


class ChainCrawler(BlobCrawler):
    def __init__(self, chain: MultiSelectChain, parser: BaseBlobParser) -> None:
        """Crawl the blob using an LLM."""
        self.chain = chain
        self.parser = parser

    def crawl(self, blob: Blob, question: str) -> List[str]:
        """Explore the blob and suggest additional content to explore."""
        records, columns = _extract_records(blob)

        result = self.chain(
            inputs={"question": question, "choices": records, "columns": columns}
        )

        selected_records = result["selected"]

        urls = [
            # TODO(): handle absolute links
            urllib.parse.urljoin(blob.source, record["link"])
            for record in selected_records
            if "mailto:" not in record["link"]
        ]
        return urls

    @classmethod
    def from_default(
        cls,
        llm: BaseLanguageModel,
        blob_parser: BaseBlobParser = MarkdownifyHTMLParser(),
    ) -> "ChainCrawler":
        """Create a crawler from the default LLM."""
        chain = MultiSelectChain.from_default(llm)
        return cls(chain=chain, parser=blob_parser)


class DocumentProcessor(abc.ABC):
    def transform(self, document: Document) -> List[Document]:
        """Transform the document."""
        raise NotImplementedError()


class ReadEntireDocChain(Chain):
    """Read entire document chain.

    This chain implements a brute force approach to reading an entire document.
    """

    chain: Chain
    text_splitter: TextSplitter
    max_num_docs: int = -1

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["doc", "question"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["document"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process a long document."""
        source_document = inputs["doc"]

        if not isinstance(source_document, Document):
            raise TypeError(f"Expected a Document, got {type(source_document)}")

        question = inputs["question"]
        sub_docs = self.text_splitter.split_documents([source_document])
        if self.max_num_docs > 0:
            _sub_docs = sub_docs[: self.max_num_docs]
        else:
            _sub_docs = sub_docs

        response = self.chain(
            {"input_documents": _sub_docs, "question": question},
            callbacks=run_manager.get_child(),
        )
        summary_doc = Document(
            page_content=response["output_text"],
            metadata=source_document.metadata,
        )
        return {"document": summary_doc}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process a long document."""
        doc = inputs["doc"]
        question = inputs["question"]
        sub_docs = self.text_splitter.split_documents([doc])
        if self.max_num_docs > 0:
            _sub_docs = sub_docs[: self.max_num_docs]
        else:
            _sub_docs = sub_docs
        results = await self.chain.acall(
            {"input_documents": _sub_docs, "question": question},
            callbacks=run_manager.get_child(),
        )
        summary_doc = Document(
            page_content=results["output_text"],
            metadata=doc.metadata,
        )

        return {"document": summary_doc}


class ParallelApply(Chain):
    """Apply a chain in parallel."""

    chain: Chain
    max_concurrency: int

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["inputs"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["results"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain."""
        # TODO(): parallelize this
        chain_inputs = inputs["inputs"]

        results = [
            self.chain(
                chain_input,
                callbacks=run_manager.get_child() if run_manager else None,
            )
            for chain_input in chain_inputs
        ]
        return {"results": results}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain."""
        chain_inputs = inputs["inputs"]

        results = await asyncio.gather(
            *[
                self.chain.acall(
                    chain_input,
                    callbacks=run_manager.get_child() if run_manager else None,
                )
                for chain_input in chain_inputs
            ]
        )
        return {"results": results}
