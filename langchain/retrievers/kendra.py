import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever


def clean_excerpt(excerpt: str) -> str:
    """Cleans an excerpt from Kendra.

    Args:
        excerpt: The excerpt to clean.

    Returns:
        The cleaned excerpt.

    """
    if not excerpt:
        return excerpt
    res = re.sub("\s+", " ", excerpt).replace("...", "")
    return res


def combined_text(title: str, excerpt: str) -> str:
    """Combines a title and an excerpt into a single string.

    Args:
        title: The title of the document.
        excerpt: The excerpt of the document.

    Returns:
        The combined text.

    """
    if not title or not excerpt:
        return ""
    return f"Document Title: {title} \nDocument Excerpt: \n{excerpt}\n"


class Highlight(BaseModel, extra=Extra.allow):
    BeginOffset: int
    EndOffset: int
    TopAnswer: Optional[bool]
    Type: Optional[str]


class TextWithHighLights(BaseModel, extra=Extra.allow):
    """Text with highlights."""

    Text: str
    """The text."""
    Highlights: Optional[Any]
    """The highlights."""


class AdditionalResultAttributeValue(BaseModel, extra=Extra.allow):
    """The value of an additional result attribute."""

    TextWithHighlightsValue: TextWithHighLights
    """The text with highlights value."""


class AdditionalResultAttribute(BaseModel, extra=Extra.allow):
    """An additional result attribute."""

    Key: str
    """The key of the attribute."""
    ValueType: Literal["TEXT_WITH_HIGHLIGHTS_VALUE"]
    """The type of the value."""
    Value: AdditionalResultAttributeValue
    """The value of the attribute."""

    def get_value_text(self) -> str:
        return self.Value.TextWithHighlightsValue.Text


class QueryResultItem(BaseModel, extra=Extra.allow):
    """A query result item."""

    DocumentId: str
    DocumentTitle: TextWithHighLights
    DocumentURI: Optional[str]
    FeedbackToken: Optional[str]
    Format: Optional[str]
    Id: Optional[str]
    Type: Optional[str]
    AdditionalAttributes: Optional[List[AdditionalResultAttribute]] = []
    DocumentExcerpt: Optional[TextWithHighLights]

    def get_attribute_value(self) -> str:
        if not self.AdditionalAttributes:
            return ""
        if not self.AdditionalAttributes[0]:
            return ""
        else:
            return self.AdditionalAttributes[0].get_value_text()

    def get_excerpt(self) -> str:
        if (
            self.AdditionalAttributes
            and self.AdditionalAttributes[0].Key == "AnswerText"
        ):
            excerpt = self.get_attribute_value()
        elif self.DocumentExcerpt:
            excerpt = self.DocumentExcerpt.Text
        else:
            excerpt = ""

        return clean_excerpt(excerpt)

    def to_doc(self) -> Document:
        title = self.DocumentTitle.Text
        source = self.DocumentURI
        excerpt = self.get_excerpt()
        type = self.Type
        page_content = combined_text(title, excerpt)
        metadata = {"source": source, "title": title, "excerpt": excerpt, "type": type}
        return Document(page_content=page_content, metadata=metadata)


class QueryResult(BaseModel, extra=Extra.allow):
    """A query result."""

    ResultItems: List[QueryResultItem]

    def get_top_k_docs(self, top_n: int) -> List[Document]:
        """Gets the top k documents.

        Args:
            top_n: The number of documents to return.

        Returns:
            The top k documents.
        """
        items_len = len(self.ResultItems)
        count = items_len if items_len < top_n else top_n
        docs = [self.ResultItems[i].to_doc() for i in range(0, count)]

        return docs


class DocumentAttributeValue(BaseModel, extra=Extra.allow):
    """The value of a document attribute."""

    DateValue: Optional[str]
    """The date value."""
    LongValue: Optional[int]
    """The long value."""
    StringListValue: Optional[List[str]]
    """The string list value."""
    StringValue: Optional[str]
    """The string value."""


class DocumentAttribute(BaseModel, extra=Extra.allow):
    """A document attribute."""

    Key: str
    """The key of the attribute."""
    Value: DocumentAttributeValue
    """The value of the attribute."""


class RetrieveResultItem(BaseModel, extra=Extra.allow):
    """A retrieve result item."""

    Content: Optional[str]
    """The content of the item."""
    DocumentAttributes: Optional[List[DocumentAttribute]] = []
    """The document attributes."""
    DocumentId: Optional[str]
    """The document ID."""
    DocumentTitle: Optional[str]
    """The document title."""
    DocumentURI: Optional[str]
    """The document URI."""
    Id: Optional[str]
    """The ID of the item."""

    def get_excerpt(self) -> str:
        if not self.Content:
            return ""
        return clean_excerpt(self.Content)

    def to_doc(self) -> Document:
        title = self.DocumentTitle if self.DocumentTitle else ""
        source = self.DocumentURI
        excerpt = self.get_excerpt()
        page_content = combined_text(title, excerpt)
        metadata = {"source": source, "title": title, "excerpt": excerpt}
        return Document(page_content=page_content, metadata=metadata)


class RetrieveResult(BaseModel, extra=Extra.allow):
    """A retrieve result."""

    QueryId: str
    """The ID of the query."""
    ResultItems: List[RetrieveResultItem]
    """The result items."""

    def get_top_k_docs(self, top_n: int) -> List[Document]:
        items_len = len(self.ResultItems)
        count = items_len if items_len < top_n else top_n
        docs = [self.ResultItems[i].to_doc() for i in range(0, count)]

        return docs


class AmazonKendraRetriever(BaseRetriever):
    """Retriever for the Amazon Kendra Index.

    Args:
        index_id: Kendra index id

        region_name: The aws region e.g., `us-west-2`.
            Fallsback to AWS_DEFAULT_REGION env variable
            or region specified in ~/.aws/config.

        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.

        top_k: No of results to return

        attribute_filter: Additional filtering of results based on metadata
            See: https://docs.aws.amazon.com/kendra/latest/APIReference

        client: boto3 client for Kendra

    Example:
        .. code-block:: python

            retriever = AmazonKendraRetriever(
                index_id="c0806df7-e76b-4bce-9b5c-d5582f6b1a03"
            )

    """

    index_id: str
    region_name: Optional[str] = None
    credentials_profile_name: Optional[str] = None
    top_k: int = 3
    attribute_filter: Optional[Dict] = None
    client: Any

    @root_validator(pre=True)
    def create_client(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("client") is not None:
            return values

        try:
            import boto3

            if values.get("credentials_profile_name"):
                session = boto3.Session(profile_name=values["credentials_profile_name"])
            else:
                # use default credentials
                session = boto3.Session()

            client_params = {}
            if values.get("region_name"):
                client_params["region_name"] = values["region_name"]

            values["client"] = session.client("kendra", **client_params)

            return values
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e

    def _kendra_query(
        self,
        query: str,
        top_k: int,
        attribute_filter: Optional[Dict] = None,
    ) -> List[Document]:
        if attribute_filter is not None:
            response = self.client.retrieve(
                IndexId=self.index_id,
                QueryText=query.strip(),
                PageSize=top_k,
                AttributeFilter=attribute_filter,
            )
        else:
            response = self.client.retrieve(
                IndexId=self.index_id, QueryText=query.strip(), PageSize=top_k
            )
        r_result = RetrieveResult.parse_obj(response)
        result_len = len(r_result.ResultItems)

        if result_len == 0:
            # retrieve API returned 0 results, call query API
            if attribute_filter is not None:
                response = self.client.query(
                    IndexId=self.index_id,
                    QueryText=query.strip(),
                    PageSize=top_k,
                    AttributeFilter=attribute_filter,
                )
            else:
                response = self.client.query(
                    IndexId=self.index_id, QueryText=query.strip(), PageSize=top_k
                )
            q_result = QueryResult.parse_obj(response)
            docs = q_result.get_top_k_docs(top_k)
        else:
            docs = r_result.get_top_k_docs(top_k)
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Run search on Kendra index and get top k documents

        Example:
        .. code-block:: python

            docs = retriever.get_relevant_documents('This is my query')

        """
        docs = self._kendra_query(query, self.top_k, self.attribute_filter)
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError("Async version is not implemented for Kendra yet.")
