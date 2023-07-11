"""Document transformers that use OpenAI Functions models"""
from typing import Any, Dict, Optional, Sequence, Type, Union

from pydantic import BaseModel, Field

from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseDocumentTransformer, Document

_METADATA_EXTRACTION_TEMPLATE = """Extract relevant information\
 from the following text:

{input}
"""


class MetadataExtractor(BaseDocumentTransformer, BaseModel):
    """Perform K-means clustering on document vectors.
    Returns an arbitrary number of documents closest to center."""

    extraction_chain: LLMChain
    """ The chain used to extract metadata from each document."""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Automatically extract and populate metadata
        for each document according to the provided schema."""

        new_documents = []

        for document in documents:
            extracted_metadata = self.extraction_chain.run(document.page_content)
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


def create_metadata_extractor(
    metadata_schema: Union[Dict[str, Any], Type[BaseModel]],
    llm: Optional[ChatOpenAI] = None,
    prompt: Optional[ChatPromptTemplate] = None,
    **extraction_chain_kwargs: Any
) -> MetadataExtractor:
    llm = llm or ChatOpenAI(
        model="gpt-3.5-turbo-0613",
    )
    prompt = prompt or ChatPromptTemplate.from_template(_METADATA_EXTRACTION_TEMPLATE)
    extraction_chain = create_structured_output_chain(
        output_schema=metadata_schema, llm=llm, prompt=prompt, **extraction_chain_kwargs
    )
    return MetadataExtractor(extraction_chain=extraction_chain)
