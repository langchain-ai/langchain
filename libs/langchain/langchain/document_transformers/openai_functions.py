"""Document transformers that use OpenAI Functions models"""
from typing import Any, Dict, Optional, Sequence, Type, Union

from pydantic import BaseModel

from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions import create_tagging_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseDocumentTransformer, BaseLanguageModel, Document


class OpenAIMetadataTagger(BaseDocumentTransformer, BaseModel):
    """Extract metadata tags from document contents using OpenAI functions.

    Example:
        .. code-block:: python

                from langchain.chat_models import ChatOpenAI
                from langchain.document_transformers import OpenAIMetadataTagger
                from langchain.schema import Document

                schema = {
                    "properties": {
                        "movie_title": { "type": "string" },
                        "critic": { "type": "string" },
                        "tone": {
                            "type": "string",
                            "enum": ["positive", "negative"]
                        },
                        "rating": {
                            "type": "integer",
                            "description": "The number of stars the critic rated the movie"
                        }
                    },
                    "required": ["movie_title", "critic", "tone"]
                }

                # Must be an OpenAI model that supports functions
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
                tagging_chain = create_tagging_chain(schema, llm)
                document_transformer = OpenAIMetadataTagger(tagging_chain=tagging_chain)
                original_documents = [
                    Document(page_content="Review of The Bee Movie\nBy Roger Ebert\n\This is the greatest movie ever made. 4 out of 5 stars."),
                    Document(page_content="Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.", metadata={"reliable": False}),
                ]

                enhanced_documents = document_transformer.transform_documents(original_documents)
    """  # noqa: E501

    tagging_chain: LLMChain
    """The chain used to extract metadata from each document."""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Automatically extract and populate metadata
        for each document according to the provided schema."""

        new_documents = []

        for document in documents:
            extracted_metadata: Dict = self.tagging_chain.run(document.page_content)  # type: ignore[assignment]  # noqa: E501
            new_document = Document(
                page_content=document.page_content,
                metadata={**extracted_metadata, **document.metadata},
            )
            new_documents.append(new_document)
        return new_documents

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError


def create_metadata_tagger(
    metadata_schema: Union[Dict[str, Any], Type[BaseModel]],
    llm: BaseLanguageModel,
    prompt: Optional[ChatPromptTemplate] = None,
    *,
    tagging_chain_kwargs: Optional[Dict] = None,
) -> OpenAIMetadataTagger:
    """Create a DocumentTransformer that uses an OpenAI function chain to automatically
        tag documents with metadata based on their content and an input schema.

    Args:
        metadata_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use, assumed to support the OpenAI function-calling API.
            Defaults to use "gpt-3.5-turbo-0613"
        prompt: BasePromptTemplate to pass to the model.

    Returns:
        An LLMChain that will pass the given function to the model.

    Example:
        .. code-block:: python

                from langchain.chat_models import ChatOpenAI
                from langchain.document_transformers import create_metadata_tagger
                from langchain.schema import Document

                schema = {
                    "properties": {
                        "movie_title": { "type": "string" },
                        "critic": { "type": "string" },
                        "tone": {
                            "type": "string",
                            "enum": ["positive", "negative"]
                        },
                        "rating": {
                            "type": "integer",
                            "description": "The number of stars the critic rated the movie"
                        }
                    },
                    "required": ["movie_title", "critic", "tone"]
                }

                # Must be an OpenAI model that supports functions
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

                document_transformer = create_metadata_tagger(schema, llm)
                original_documents = [
                    Document(page_content="Review of The Bee Movie\nBy Roger Ebert\n\This is the greatest movie ever made. 4 out of 5 stars."),
                    Document(page_content="Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.", metadata={"reliable": False}),
                ]

                enhanced_documents = document_transformer.transform_documents(original_documents)
    """  # noqa: E501
    metadata_schema = (
        metadata_schema
        if isinstance(metadata_schema, dict)
        else metadata_schema.schema()
    )
    _tagging_chain_kwargs = tagging_chain_kwargs or {}
    tagging_chain = create_tagging_chain(
        metadata_schema, llm, prompt=prompt, **_tagging_chain_kwargs
    )
    return OpenAIMetadataTagger(tagging_chain=tagging_chain)
