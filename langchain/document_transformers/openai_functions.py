"""Document transformers that use OpenAI Functions models"""
from typing import Any, Dict, Optional, Sequence, Type, Union

from pydantic import BaseModel, Field

from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions import (
    create_tagging_chain,
    create_tagging_chain_pydantic,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseDocumentTransformer, Document


class MetadataTagger(BaseDocumentTransformer, BaseModel):
    """Perform K-means clustering on document vectors.
    Returns an arbitrary number of documents closest to center."""

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
    llm = llm or ChatOpenAI(
        model="gpt-3.5-turbo-0613",
    )
    metadata_schema = metadata_schema if isinstance(metadata_schema, dict) else metadata_schema.schema()
    tagging_chain = create_tagging_chain(
        schema=metadata_schema, llm=llm, prompt=prompt, **tagging_chain_kwargs
    )
    return MetadataTagger(tagging_chain=tagging_chain)
