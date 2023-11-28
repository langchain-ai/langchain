import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator, validator
from langchain_core.retrievers import BaseRetriever

from langchain.callbacks.manager import CallbackManagerForRetrieverRun


def clean_excerpt(excerpt: str) -> str:
    """Clean an excerpt from Kendra.

    Args:
        excerpt: The excerpt to clean.

    Returns:
        The cleaned excerpt.

    """
    if not excerpt:
        return excerpt
    res = re.sub(r"\s+", " ", excerpt).replace("...", "")
    return res


def combined_text(item: "ResultItem") -> str:
    """Combine a ResultItem title and excerpt into a single string.

    Args:
        item: the ResultItem of a Kendra search.

    Returns:
        A combined text of the title and excerpt of the given item.

    """
    text = ""
    title = item.get_title()
    if title:
        text += f"Document Title: {title}\n"
    excerpt = clean_excerpt(item.get_excerpt())
    if excerpt:
        text += f"Document Excerpt: \n{excerpt}\n"
    return text


DocumentAttributeValueType = Union[str, int, List[str], None]
"""Possible types of a DocumentAttributeValue.

Dates are also represented as str.
"""


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class Highlight(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Information that highlights the keywords in the excerpt."""

    BeginOffset: int
    """The zero-based location in the excerpt where the highlight starts."""
    EndOffset: int
    """The zero-based location in the excerpt where the highlight ends."""
    TopAnswer: Optional[bool]
    """Indicates whether the result is the best one."""
    Type: Optional[str]
    """The highlight type: STANDARD or THESAURUS_SYNONYM."""


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class TextWithHighLights(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Text with highlights."""

    Text: str
    """The text."""
    Highlights: Optional[Any]
    """The highlights."""


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class AdditionalResultAttributeValue(  # type: ignore[call-arg]
    BaseModel, extra=Extra.allow
):
    """Value of an additional result attribute."""

    TextWithHighlightsValue: TextWithHighLights
    """The text with highlights value."""


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class AdditionalResultAttribute(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Additional result attribute."""

    Key: str
    """The key of the attribute."""
    ValueType: Literal["TEXT_WITH_HIGHLIGHTS_VALUE"]
    """The type of the value."""
    Value: AdditionalResultAttributeValue
    """The value of the attribute."""

    def get_value_text(self) -> str:
        return self.Value.TextWithHighlightsValue.Text


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class DocumentAttributeValue(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Value of a document attribute."""

    DateValue: Optional[str]
    """The date expressed as an ISO 8601 string."""
    LongValue: Optional[int]
    """The long value."""
    StringListValue: Optional[List[str]]
    """The string list value."""
    StringValue: Optional[str]
    """The string value."""

    @property
    def value(self) -> DocumentAttributeValueType:
        """The only defined document attribute value or None.
        According to Amazon Kendra, you can only provide one
        value for a document attribute.
        """
        if self.DateValue:
            return self.DateValue
        if self.LongValue:
            return self.LongValue
        if self.StringListValue:
            return self.StringListValue
        if self.StringValue:
            return self.StringValue

        return None


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class DocumentAttribute(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Document attribute."""

    Key: str
    """The key of the attribute."""
    Value: DocumentAttributeValue
    """The value of the attribute."""


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class ResultItem(BaseModel, ABC, extra=Extra.allow):  # type: ignore[call-arg]
    """Base class of a result item."""

    Id: Optional[str]
    """The ID of the relevant result item."""
    DocumentId: Optional[str]
    """The document ID."""
    DocumentURI: Optional[str]
    """The document URI."""
    DocumentAttributes: Optional[List[DocumentAttribute]] = []
    """The document attributes."""

    @abstractmethod
    def get_title(self) -> str:
        """Document title."""

    @abstractmethod
    def get_excerpt(self) -> str:
        """Document excerpt or passage original content as retrieved by Kendra."""

    def get_additional_metadata(self) -> dict:
        """Document additional metadata dict.
        This returns any extra metadata except these:
            * result_id
            * document_id
            * source
            * title
            * excerpt
            * document_attributes
        """
        return {}

    def get_document_attributes_dict(self) -> Dict[str, DocumentAttributeValueType]:
        """Document attributes dict."""
        return {attr.Key: attr.Value.value for attr in (self.DocumentAttributes or [])}

    def to_doc(
        self, page_content_formatter: Callable[["ResultItem"], str] = combined_text
    ) -> Document:
        """Converts this item to a Document."""
        page_content = page_content_formatter(self)
        metadata = self.get_additional_metadata()
        metadata.update(
            {
                "result_id": self.Id,
                "document_id": self.DocumentId,
                "source": self.DocumentURI,
                "title": self.get_title(),
                "excerpt": self.get_excerpt(),
                "document_attributes": self.get_document_attributes_dict(),
            }
        )

        return Document(page_content=page_content, metadata=metadata)


class QueryResultItem(ResultItem):
    """Query API result item."""

    DocumentTitle: TextWithHighLights
    """The document title."""
    FeedbackToken: Optional[str]
    """Identifies a particular result from a particular query."""
    Format: Optional[str]
    """
    If the Type is ANSWER, then format is either:
        * TABLE: a table excerpt is returned in TableExcerpt;
        * TEXT: a text excerpt is returned in DocumentExcerpt.
    """
    Type: Optional[str]
    """Type of result: DOCUMENT or QUESTION_ANSWER or ANSWER"""
    AdditionalAttributes: Optional[List[AdditionalResultAttribute]] = []
    """One or more additional attributes associated with the result."""
    DocumentExcerpt: Optional[TextWithHighLights]
    """Excerpt of the document text."""

    def get_title(self) -> str:
        return self.DocumentTitle.Text

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

        return excerpt

    def get_additional_metadata(self) -> dict:
        additional_metadata = {"type": self.Type}
        return additional_metadata


class RetrieveResultItem(ResultItem):
    """Retrieve API result item."""

    DocumentTitle: Optional[str]
    """The document title."""
    Content: Optional[str]
    """The content of the item."""

    def get_title(self) -> str:
        return self.DocumentTitle or ""

    def get_excerpt(self) -> str:
        return self.Content or ""


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class QueryResult(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """`Amazon Kendra Query API` search result.

    It is composed of:
        * Relevant suggested answers: either a text excerpt or table excerpt.
        * Matching FAQs or questions-answer from your FAQ file.
        * Documents including an excerpt of each document with its title.
    """

    ResultItems: List[QueryResultItem]
    """The result items."""


# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class RetrieveResult(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """`Amazon Kendra Retrieve API` search result.

    It is composed of:
        * relevant passages or text excerpts given an input query.
    """

    QueryId: str
    """The ID of the query."""
    ResultItems: List[RetrieveResultItem]
    """The result items."""


class AmazonKendraRetriever(BaseRetriever):
    """`Amazon Kendra Index` retriever.

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

        page_content_formatter: generates the Document page_content
            allowing access to all result item attributes. By default, it uses
            the item's title and excerpt.

        client: boto3 client for Kendra

        user_context: Provides information about the user context
            See: https://docs.aws.amazon.com/kendra/latest/APIReference

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
    page_content_formatter: Callable[[ResultItem], str] = combined_text
    client: Any
    user_context: Optional[Dict] = None

    @validator("top_k")
    def validate_top_k(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"top_k ({value}) cannot be negative.")
        return value

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

    def _kendra_query(self, query: str) -> Sequence[ResultItem]:
        kendra_kwargs = {
            "IndexId": self.index_id,
            "QueryText": query.strip(),
            "PageSize": self.top_k,
        }
        if self.attribute_filter is not None:
            kendra_kwargs["AttributeFilter"] = self.attribute_filter
        if self.user_context is not None:
            kendra_kwargs["UserContext"] = self.user_context

        response = self.client.retrieve(**kendra_kwargs)
        r_result = RetrieveResult.parse_obj(response)
        if r_result.ResultItems:
            return r_result.ResultItems

        # Retrieve API returned 0 results, fall back to Query API
        response = self.client.query(**kendra_kwargs)
        q_result = QueryResult.parse_obj(response)
        return q_result.ResultItems

    def _get_top_k_docs(self, result_items: Sequence[ResultItem]) -> List[Document]:
        top_docs = [
            item.to_doc(self.page_content_formatter)
            for item in result_items[: self.top_k]
        ]
        return top_docs

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
        result_items = self._kendra_query(query)
        top_k_docs = self._get_top_k_docs(result_items)
        return top_k_docs
