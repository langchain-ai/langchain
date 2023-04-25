import json
from typing import Any, Dict, List, cast

from pydantic import BaseModel, Field, root_validator

from langchain import LLMChain, PromptTemplate
from langchain.agents.agent_toolkits import VectorStoreInfo
from langchain.llms import BaseLLM
from langchain.retrievers.pinecone_self_query_prompt import (
    PineconeSelfQueryOutputParser,
    pinecone_example,
    pinecone_format_instructions,
    self_query_prompt,
)
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import Pinecone


class MetadataFieldInfo(BaseModel):
    """Information about a vectorstore metadata field."""

    name: str
    description: str
    examples: List
    type: str

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class VectorStoreExtendedInfo(VectorStoreInfo):
    """Extension of VectorStoreInfo that includes info about metadata fields."""

    metadata_field_info: List[MetadataFieldInfo]
    """Map of metadata field name to info about that field."""


def _format_metadata_field_info(info: List[MetadataFieldInfo]) -> str:
    info_dicts = {}
    for i in info:
        i_dict = dict(i)
        info_dicts[i_dict.pop("name")] = i_dict
    return json.dumps(info_dicts, indent=2).replace("{", "{{").replace("}", "}}")


class PineconeSelfQueryRetriever(BaseRetriever, BaseModel):
    """Retriever that wraps around a Pinecone vector store and uses an LLM to generate
    the vector store queries."""

    vectorstore: Pinecone
    """The Pinecone vector store from which documents will be retrieved."""
    llm_chain: LLMChain
    """The LLMChain for generating the vector store queries."""
    search_type: str = "similarity"
    """The search type to perform on the vector store."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass in to the vector store search."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "mmr"):
                raise ValueError(
                    f"search_type of {search_type} not allowed. Expected "
                    "search_type to be 'similarity' or 'mmr'."
                )
        return values

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        inputs = self.llm_chain.prep_inputs(query)
        vectorstore_query = cast(dict, self.llm_chain.predict_and_parse(**inputs))
        print(vectorstore_query)
        new_query = vectorstore_query["search_string"]
        _filter = vectorstore_query["metadata_filter"]
        docs = self.vectorstore.search(
            new_query, self.search_type, filter=_filter, **self.search_kwargs
        )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        inputs = self.llm_chain.prep_inputs(query)
        vectorstore_query = cast(dict, self.llm_chain.apredict_and_parse(**inputs))
        new_query = vectorstore_query["query"]
        _filter = vectorstore_query["filter"]
        docs = await self.vectorstore.asearch(
            new_query, self.search_type, filter=_filter, **self.search_kwargs
        )
        return docs

    @classmethod
    def from_vectorstore_info(
        cls,
        llm: BaseLLM,
        vectorstore_info: VectorStoreExtendedInfo,
        **kwargs: Any,
    ) -> "PineconeSelfQueryRetriever":
        metadata_field_json = _format_metadata_field_info(
            vectorstore_info.metadata_field_info
        )
        prompt_str = self_query_prompt.format(
            format_instructions=pinecone_format_instructions,
            example=pinecone_example,
            docstore_description=vectorstore_info.description,
            metadata_fields=metadata_field_json,
        )
        prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt_str,
            output_parser=PineconeSelfQueryOutputParser(),
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            llm_chain=llm_chain,
            vectorstore=vectorstore_info.vectorstore,
            **kwargs,
        )
