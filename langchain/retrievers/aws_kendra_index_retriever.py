"""Retriever wrapper for AWS Kendra."""
import re
from typing import Any, Dict, List

from langchain.schema import BaseRetriever, Document


class AwsKendraIndexRetriever(BaseRetriever):
    """Wrapper around AWS Kendra."""

    kendraindex: str
    """Kendra index id"""
    k: int
    """Number of documents to query for."""
    languagecode: str
    """Languagecode used for querying."""
    kclient: Any
    """ boto3 client for Kendra. """

    def __init__(
        self, kclient: Any, kendraindex: str, k: int = 3, languagecode: str = "en"
    ):
        self.kendraindex = kendraindex
        self.k = k
        self.languagecode = languagecode
        self.kclient = kclient

    def _clean_result(self, res_text: str) -> str:
        return re.sub("\s+", " ", res_text).replace("...", "")

    def _get_top_n_results(self, resp: Dict, count: int) -> Document:
        r = resp["ResultItems"][count]
        doc_title = r["DocumentTitle"]["Text"]
        doc_uri = r["DocumentURI"]
        r_type = r["Type"]

        if (
            r["AdditionalAttributes"]
            and r["AdditionalAttributes"][0]["Key"] == "AnswerText"
        ):
            res_text = r["AdditionalAttributes"][0]["Value"]["TextWithHighlightsValue"][
                "Text"
            ]
        else:
            res_text = r["DocumentExcerpt"]["Text"]

        doc_excerpt = self._clean_result(res_text)
        combined_text = f"""Document Title: {doc_title}
Document Excerpt: {doc_excerpt}
"""

        return Document(
            page_content=combined_text,
            metadata={
                "source": doc_uri,
                "title": doc_title,
                "excerpt": doc_excerpt,
                "type": r_type,
            },
        )

    def _kendra_query(self, kquery: str) -> List[Document]:
        response = self.kclient.query(
            IndexId=self.kendraindex,
            QueryText=kquery.strip(),
            AttributeFilter={
                "AndAllFilters": [
                    {
                        "EqualsTo": {
                            "Key": "_language_code",
                            "Value": {
                                "StringValue": self.languagecode,
                            },
                        }
                    }
                ]
            },
        )

        if len(response["ResultItems"]) > self.k:
            r_count = self.k
        else:
            r_count = len(response["ResultItems"])

        return [self._get_top_n_results(response, i) for i in range(0, r_count)]

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Run search on Kendra index and get top k documents

        docs = get_relevant_documents('This is my query')
        """
        return self._kendra_query(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("AwsKendraIndexRetriever does not support async")
