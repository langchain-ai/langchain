"""Document transformers that use OpenAI Functions models"""
from typing import Any, Dict, Optional, Sequence, Type, Union

from pydantic import BaseModel

from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions import create_tagging_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseDocumentTransformer, Document


class MetadataTagger(BaseDocumentTransformer, BaseModel):
    tagging_chain: LLMChain
    """ The chain used to extract metadata from each document."""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Automatically extract and populate metadata
        for each document according to the provided schema."""

        new_documents = []

        for document in documents:
            extracted_metadata = self.tagging_chain.run(document.page_content)
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
    llm: Optional[ChatOpenAI] = None,
    prompt: Optional[ChatPromptTemplate] = None,
    **tagging_chain_kwargs: Any
) -> MetadataTagger:
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

                from langchain.docstore.document import Document
                from langchain.chat_models import ChatOpenAI
                from langchain.document_transformers.openai_functions import create_metadata_tagger

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

                document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
                original_documents = [
                    Document(page_content="Review of The Bee Movie\nBy Roger Ebert\n\This is the greatest movie ever made. 4 out of 5 stars."),
                    Document(page_content="Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.", metadata={"reliable": False}),
                ]

                enhanced_documents = document_transformer.transform_documents(documents=original_documents)
                print(*enhanced_documents, sep="\n")
                # -> page_content='Review of The Bee Movie\nBy Roger Ebert\n\\This is the greatest movie ever made. 4 out of 5 stars.' metadata={'movie_title': 'The Bee Movie', 'critic': 'Roger Ebert', 'tone': 'positive', 'rating': 4}
                # -> page_content='Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.' metadata={'movie_title': 'The Godfather', 'critic': 'Anonymous', 'tone': 'negative', 'rating': 1, 'reliable': False}
    """  # noqa: E501
    llm = llm or ChatOpenAI(
        model="gpt-3.5-turbo-0613",
    )
    metadata_schema = (
        metadata_schema
        if isinstance(metadata_schema, dict)
        else metadata_schema.schema()
    )
    tagging_chain = create_tagging_chain(
        schema=metadata_schema, llm=llm, prompt=prompt, **tagging_chain_kwargs
    )
    return MetadataTagger(tagging_chain=tagging_chain)
