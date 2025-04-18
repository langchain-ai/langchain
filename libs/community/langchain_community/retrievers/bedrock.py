from typing import Any, Dict, List, Optional

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, model_validator


class VectorSearchConfig(BaseModel, extra="allow"):
    """Configuration for vector search."""

    numberOfResults: int = 4


class RetrievalConfig(BaseModel, extra="allow"):
    """Configuration for retrieval."""

    vectorSearchConfiguration: VectorSearchConfig


@deprecated(
    since="0.3.16",
    removal="1.0",
    alternative_import="langchain_aws.AmazonKnowledgeBasesRetriever",
)
class AmazonKnowledgeBasesRetriever(BaseRetriever):
    """Amazon Bedrock Knowledge Bases retriever.

    See https://aws.amazon.com/bedrock/knowledge-bases for more info.

    Setup:
        Install ``langchain-aws``:

        .. code-block:: bash

            pip install -U langchain-aws

    Key init args:
        knowledge_base_id: Knowledge Base ID.
        region_name: The aws region e.g., `us-west-2`.
            Fallback to AWS_DEFAULT_REGION env variable or region specified in
            ~/.aws/config.
        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.
        client: boto3 client for bedrock agent runtime.
        retrieval_config: Configuration for retrieval.

    Instantiate:
        .. code-block:: python

            from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id="<knowledge-base-id>",
                retrieval_config={
                    "vectorSearchConfiguration": {
                        "numberOfResults": 4
                    }
                },
            )

    Usage:
        .. code-block:: python

            query = "..."

            retriever.invoke(query)

    Use within a chain:
        .. code-block:: python

            from langchain_aws import ChatBedrockConverse
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatBedrockConverse(
                model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
            )

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("...")

    """  # noqa: E501

    knowledge_base_id: str
    region_name: Optional[str] = None
    credentials_profile_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    client: Any
    retrieval_config: RetrievalConfig

    @model_validator(mode="before")
    @classmethod
    def create_client(cls, values: Dict[str, Any]) -> Any:
        if values.get("client") is not None:
            return values

        try:
            import boto3
            from botocore.client import Config
            from botocore.exceptions import UnknownServiceError

            if values.get("credentials_profile_name"):
                session = boto3.Session(profile_name=values["credentials_profile_name"])
            else:
                # use default credentials
                session = boto3.Session()

            client_params = {
                "config": Config(
                    connect_timeout=120, read_timeout=120, retries={"max_attempts": 0}
                )
            }
            if values.get("region_name"):
                client_params["region_name"] = values["region_name"]

            if values.get("endpoint_url"):
                client_params["endpoint_url"] = values["endpoint_url"]

            values["client"] = session.client("bedrock-agent-runtime", **client_params)

            return values
        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except UnknownServiceError as e:
            raise ImportError(
                "Ensure that you have installed the latest boto3 package "
                "that contains the API for `bedrock-runtime-agent`."
            ) from e
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        response = self.client.retrieve(
            retrievalQuery={"text": query.strip()},
            knowledgeBaseId=self.knowledge_base_id,
            retrievalConfiguration=self.retrieval_config.dict(),
        )
        results = response["retrievalResults"]
        documents = []
        for result in results:
            content = result["content"]["text"]
            result.pop("content")
            if "score" not in result:
                result["score"] = 0
            if "metadata" in result:
                result["source_metadata"] = result.pop("metadata")
            documents.append(
                Document(
                    page_content=content,
                    metadata=result,
                )
            )

        return documents
