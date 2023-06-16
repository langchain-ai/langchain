import json
import logging
import time
import urllib.error
import urllib.request
from typing import List

from pydantic import BaseModel, Extra

from langchain.schema import Document

logger = logging.getLogger(__name__)


class PubMedAPIWrapper():
    """
    Wrapper around PubMed API (https://www.ncbi.nlm.nih.gov/books/NBK25501/).

    This wrapper will use the PubMed API to conduct searches and fetch
    document summaries. By default, it will return the document summaries
    of the top-k results of an input search.
    """

    base_url_esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    base_url_efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    load_all_available_meta: bool = False
    doc_content_chars_max: int = 2000
    retmax = 500  # The number of articles fetched each time
    max_retry = 5
    sleep_time = 0.2

    def __init__(
            self,
            api_key: str = None,
            top_k_results: int = 3,
            sort_by: str = "relevance",
            load_all_available_meta: bool = False):
        """
        Parameters:
            api_key: Users can obtain an API key now from the Settings page of their NCBI account 
                (to create an account, visit http://www.ncbi.nlm.nih.gov/account/). 
                With an API, the request limit is 10 times per minute; without it, 
                the limit is 3 times per minute.
            top_k_results: number of the top-scored document used for the PubMed tool.
            sort_by: the sort order of the results. pub_date, Author,JournalName, or relevance (default).
        """
        self.top_k_results = top_k_results
        self.sort_by = sort_by
        self.load_all_available_meta = load_all_available_meta
        if api_key is not None:
            self.base_url_esearch = self.base_url_esearch + "api_key=" + api_key
            self.base_url_efetch = self.base_url_efetch + "api_key=" + api_key

    def run(self, query: str, output_type: str = "string") -> Any:
        """
        Run PubMed search and get the article meta information.
        Parameters:
            query: the query string
            output_type: the output type of the results, list or string (default).
        """

        try:
            results = self.fetch_article(query)
            if results:
                if output_type == "string":
                    docs = [
                        f"Published: {result['pub_date']}\nTitle: {result['title']}\n"
                        f"Summary: {result['abstract']}"
                        for result in results]
                    return ("\n\n".join(docs)[: self.doc_content_chars_max])

                if output_type == "list":
                    return results

            else:
                return "No good PubMed Result was found"

        except Exception as ex:
            return f"PubMed exception: {ex}"

    def search_article(self, query: str) -> Any:
        """
        Search PubMed for documents matching the query.
        Return the counts of matched articles, the list of uids, the webenv, and the query_key.
        Parameters:
            query: the query string.
        """
        url = (
            self.base_url_esearch
            + "&db=pubmed&term="
            + str({urllib.parse.quote(query)})
            + "&retmode=json&usehistory=y"
            + "&sort="
            + self.sort_by
        )
        # print("Processing url: ", url)
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)

        count = int(json_text["esearchresult"]["count"])
        uid_list = json_text["esearchresult"]["idlist"]
        webenv = json_text["esearchresult"]["webenv"]
        query_key = json_text["esearchresult"]["querykey"]

        print("Total number of articles searched: ", count)
        return count, uid_list, webenv, query_key

    def fetch_article(self, query) -> List[dict]:
        """
        Fetch the article according to the History server.
        Return a list of dictionaries containing the document metadata.
        Parameters:
            query: the query string.
        """

        count, uid_list, webenv, query_key = self.search_article(query)

        if count < self.top_k_results:
            self.top_k_results = count

        print(f"The top {self.top_k_results} results will be processed.")

        if self.top_k_results <= self.retmax:
            self.retmax = self.top_k_results

        articles = []
        for retstart in range(0, self.top_k_results, self.retmax):
            url = (
                self.base_url_efetch
                + "&db=pubmed&retmode=xml"
                + "&webenv="
                + webenv
                + "&query_key="
                + query_key
                + "&retmax="
                + str(self.retmax)
                + "&retstart="
                + str(retstart)
            )
            # print("Processing url: ", url)
            retry = 0
            while True:
                try:
                    result = urllib.request.urlopen(url)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and retry < self.max_retry:
                        # Too Many Requests error
                        # wait for an exponentially increasing amount of time
                        print(
                            f"Too Many Requests, "
                            f"waiting for {self.sleep_time:.2f} seconds..."
                        )
                        time.sleep(self.sleep_time)
                        self.sleep_time *= 2
                        retry += 1
                    else:
                        raise e

            xml_text = result.read().decode("utf-8")
            xml_list = xml_text.split("</PubmedArticle>")[0:-1]

            for xml in xml_list:
                # Get uid
                uid = ""
                if '<PMID Version="1">' in xml and "</PMID>" in xml:
                    start_tag = '<PMID Version="1">'
                    end_tag = "</PMID>"
                    uid = xml[
                        xml.index(start_tag) + len(start_tag): xml.index(end_tag)
                    ]
                # Get title
                title = ""
                if "<ArticleTitle>" in xml and "</ArticleTitle>" in xml:
                    start_tag = "<ArticleTitle>"
                    end_tag = "</ArticleTitle>"
                    title = xml[
                        xml.index(start_tag) + len(start_tag): xml.index(end_tag)
                    ]

                # Get abstract
                abstract = ""
                if "<AbstractText>" in xml and "</AbstractText>" in xml:
                    start_tag = "<AbstractText>"
                    end_tag = "</AbstractText>"
                    abstract = xml[
                        xml.index(start_tag) + len(start_tag): xml.index(end_tag)
                    ]

                # Get publication date
                pub_date = ""
                if "<PubDate>" in xml and "</PubDate>" in xml:
                    start_tag = "<PubDate>"
                    end_tag = "</PubDate>"
                    pub_date = xml[
                        xml.index(start_tag) + len(start_tag): xml.index(end_tag)
                    ]

                # Return article as dictionary
                article = {
                    "uid": uid,
                    "title": title,
                    "abstract": abstract,
                    "pub_date": pub_date,
                }
                articles.append(article)

            per = (retstart + self.retmax) * 100 / self.top_k_results
            if per > 100:
                per = 100
            print(f"{per:.2f}% done")

        return articles[: self.top_k_results]

    def _transform_doc(self, doc: dict) -> Document:
        summary = doc.pop("abstract")
        return Document(page_content=summary, metadata=doc)

    def load_docs(self, query: str) -> List[Document]:
        document_dicts = self.fetch_article(query=query)
        return [self._transform_doc(d) for d in document_dicts]
