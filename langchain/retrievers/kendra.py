import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Extra

from langchain.docstore.document import Document
from langchain.schema import BaseRetriever


def clean_excerpt(res_text):
    res = re.sub("\s+", " ", res_text).replace("...", "")
    return res


def combined_text(title: str, excerpt: str) -> str:
    return f"Document Title: {title} \nDocument Excerpt: \n{excerpt}\n"


class Highlight(BaseModel, extra=Extra.allow):
    BeginOffset: int
    EndOffset: int
    TopAnswer: Optional[bool]
    Type: Optional[str]


class TextWithHighLights(BaseModel, extra=Extra.allow):
    Text: str
    Highlights: Optional[Any]


class AdditionalResultAttribute(BaseModel, extra=Extra.allow):
    Key: str
    ValueType: Literal["TEXT_WITH_HIGHLIGHTS_VALUE"]
    Value: Optional[TextWithHighLights]


class QueryResultItem(BaseModel, extra=Extra.allow):
    DocumentId: str
    DocumentTitle: TextWithHighLights
    DocumentURI: Optional[str]
    FeedbackToken: Optional[str]
    Format: Optional[str]
    Id: Optional[str]
    Type: Optional[str]
    AdditionalAttributes: Optional[List[AdditionalResultAttribute]] = []
    DocumentExcerpt: Optional[TextWithHighLights]

    def get_excerpt(self) -> str:
        if (
            self.AdditionalAttributes
            and self.AdditionalAttributes[0].Key == "AnswerText"
        ):
            excerpt = self.AdditionalAttributes[0].Value.Text
        else:
            excerpt = self.DocumentExcerpt.Text

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
    ResultItems: List[QueryResultItem]

    def get_top_k_docs(self, top_n: int) -> List[Document]:
        items_len = len(self.ResultItems)
        count = items_len if items_len < top_n else top_n
        docs = [self.ResultItems[i].to_doc() for i in range(0, count)]

        return docs


class DocumentAttributeValue(BaseModel, extra=Extra.allow):
    DateValue: Optional[str]
    LongValue: Optional[int]
    StringListValue: Optional[List[str]]
    StringValue: Optional[str]


class DocumentAttribute(BaseModel, extra=Extra.allow):
    Key: str
    Value: DocumentAttributeValue


class RetrieveResultItem(BaseModel, extra=Extra.allow):
    Content: Optional[str]
    DocumentAttributes: Optional[List[DocumentAttribute]] = []
    DocumentId: Optional[str]
    DocumentTitle: Optional[str]
    DocumentURI: Optional[str]
    Id: Optional[str]

    def get_excerpt(self):
        return clean_excerpt(self.Content)

    def to_doc(self) -> Document:
        title = self.DocumentTitle
        source = self.DocumentURI
        excerpt = self.get_excerpt()
        page_content = combined_text(title, excerpt)
        metadata = {"source": source, "title": title, "excerpt": excerpt}
        return Document(page_content=page_content, metadata=metadata)


class RetrieveResult(BaseModel, extra=Extra.allow):
    QueryId: str
    ResultItems: List[RetrieveResultItem]

    def get_top_k_docs(self, top_n: int) -> List[Document]:
        items_len = len(self.ResultItems)
        count = items_len if items_len < top_n else top_n
        docs = [self.ResultItems[i].to_doc() for i in range(0, count)]

        return docs


class AmazonKendraRetriever(BaseRetriever):
    """Retriever to query documents from Amazon Kendra Index.

    Example:
        .. code-block:: python

            retriever = AmazonKendraRetriever(
                index_id="c0806df7-e76b-4bce-9b5c-d5582f6b1a03"
            )

    """

    index_id: str
    """Kendra index id"""

    region_name: Optional[str] = None
    """The aws region e.g., `us-west-2`. Fallsback to AWS_DEFAULT_REGION env variable
    or region specified in ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    language_code: Optional[str] = None
    """Language code that the documents are indexed with.
    See: https://docs.aws.amazon.com/kendra/latest/dg/in-adding-languages.html
    """

    client: Optional[Any] = None
    """boto3 client for Kendra."""

    def __init__(
        self,
        index_id,
        region_name=None,
        credentials_profile_name=None,
        language_code=None,
        client=None,
    ):
        self.index_id = index_id
        self.language_code = language_code

        if client is not None:
            self.client = client
            return

        try:
            import boto3

            if self.credentials_profile_name is not None:
                session = boto3.Session(profile_name=credentials_profile_name)
            else:
                # use default credentials
                session = boto3.Session()

            client_params = {}
            if region_name is not None:
                client_params["region_name"] = region_name

            self.client = session.client("kendra", **client_params)
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

    def _kendra_query(self, query, top_k: int = 3) -> List[Document]:
        if self.language_code is not None:
            attribute_filter = {
                "AndAllFilters": [
                    {
                        "EqualsTo": {
                            "Key": "_language_code",
                            "Value": {
                                "StringValue": self.language_code,
                            },
                        }
                    }
                ]
            }

            response = self.client.retrieve(
                IndexId=self.index_id,
                QueryText=query.strip(),
                PageSize=top_k,
                AttributeFilter=attribute_filter,
            )
        else:
            response = self.client.retrieve(
                IndexId=self.index_id,
                QueryText=query.strip(),
                PageSize=top_k
            )
        r_result = RetrieveResult.parse_obj(response)
        result_len = len(r_result.ResultItems)

        if result_len == 0:
            # retrieve API returned 0 results, call query API
            if self.language_code is not None:
                response = self.client.query(
                    IndexId=self.index_id,
                    QueryText=query.strip(),
                    PageSize=top_k,
                    AttributeFilter=attribute_filter,
                )
            else:
                response = self.client.query(
                    IndexId=self.index_id,
                    QueryText=query.strip(),
                    PageSize=top_k
                )
            q_result = QueryResult.parse_obj(response)
            docs = q_result.get_top_k_docs(top_k)
        else:
            docs = r_result.get_top_k_docs(top_k)
        return docs

    def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Document]:
        """Run search on Kendra index and get top k documents

        Example:
        .. code-block:: python

            docs = retriever.get_relevant_documents('This is my query')

        """
        docs = self._kendra_query(query, top_k)
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async version is not implemented for Kendra yet.")
