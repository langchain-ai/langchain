import json
from typing import Any, Dict, List, cast

from pydantic import BaseModel, Field, root_validator

from langchain import LLMChain, PromptTemplate
from langchain.agents.agent_toolkits import VectorStoreInfo
from langchain.llms import BaseLLM
from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseOutputParser, BaseRetriever, Document
from langchain.vectorstores import VectorStore

pinecone_format_instructions = """RESPONSE FORMAT
----------------------------

When responding use a markdown code snippet formatted in the following schema:

```json
{{
    "search_string": string \\ The string to compare to document contents,
    "metadata_filter": {{
        "<MetadataField>": {{
            <Operator>: <Value>
        }},
        "<MetadataField>": {{
            <Operator>: <Value>
        }},
    }} \\ The conditions on which to filter the metadata
}}
```
Filtering metadata supports the following operators:
    $eq - Equal to (number, string, boolean)
    $ne - Not equal to (number, string, boolean)
    $gt - Greater than (number)
    $gte - Greater than or equal to (number)
    $lt - Less than (number)
    $lte - Less than or equal to (number)
    $in - In array (string or number)
    $nin - Not in array (string or number)

NOTE that if you are not exactly sure how some string metadata values are formatted you can include multiple potentially matching values. For example if you're not sure how a string value is capitalized, you can check for equality with both the fully upper-cased or fully lower-cased versions of the value.

PLEASE REMEMBER that for some queries no metadata filters are needed. In these cases you should leave "metadata_filter" as an empty map."""

pinecone_example = """EXAMPLE
----------------------------

DOCUMENT STORE DESCRIPTION: News headlines from around the world
METADATA FIELDS: {{
    "year": {{
        "description": "The year the headline was published",
        "type": "integer",
        "example_values": [2022, 1997]
    }},
    "country": {{
        "description": "The country of origin of the news media outlet",
        "type": "string",
        "example_values": ["Chile", "Japan", "Ghana"]
    }},
    "source": {{
        "description": "The name of the news media outlet",
        "type": "string",
        "example_values": ["Wall Street Journal", "New York Times", "Axios"]
    }}
}}
QUESTION: What was the sentiment of Chinese and Taiwanese media outlets regarding the 2024 trade deal between America and India?
DOCUMENT STORE QUERY:
```json
{{
    "search_string": "Trade deal between America and India",
    "metadata_filter": {{
        "country": {{
            "$in": ["China", "People's Republic of China", "PRC", "Taiwan", "Republic of China", "ROC", "china", "taiwan"]
        }},
        "year": {{
            "$gte": 2024
        }}
    }}
}}
```"""

self_query_prompt = """INSTRUCTIONS
----------------------------

You have access to a store of documents. Each document contains text and a key-value store of associated metadata. Given a user question, your job is to come up with a fully formed query to the document store that will return the most relevant documents.

A document store query consists of two components: a search string and a metadata filter. The search string is compared to the text contents of the stored documents. The metadata filter is used to filter out documents whose metadata does not match the given criteria.


{format_instructions}


{example}

Begin!

DOCUMENT STORE DESCRIPTION: {docstore_description}
METADATA FIELDS: {metadata_fields}
QUESTION: {{question}}
DOCUMENT STORE QUERY:"""


class PineconeSelfQueryOutputParser(BaseOutputParser[Dict]):
    def get_format_instructions(self) -> str:
        return pinecone_format_instructions

    def parse(self, text: str) -> Dict:
        expected_keys = ["search_string", "metadata_filter"]
        parsed = parse_json_markdown(text, expected_keys)
        if len(parsed["search_string"]) == 0:
            parsed["search_string"] = " "
        return parsed

    @property
    def _type(self) -> str:
        return "pinecone_self_query_output_parser"


class VectorStoreExtendedInfo(VectorStoreInfo):
    """Extension of VectorStoreInfo that includes info about metadata fields."""

    metadata_field_info: Dict[str, Dict]
    """Map of metadata field name to info about that field."""


class PineconeSelfQueryRetriever(BaseRetriever, BaseModel):
    vectorstore: VectorStore
    """"""
    llm_chain: LLMChain
    """"""
    search_type: str = "similarity"
    """"""
    search_kwargs: dict = Field(default_factory=dict)
    """"""

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
        metadata_field_json = (
            json.dumps(vectorstore_info.metadata_field_info, indent=2)
            .replace("{", "{{")
            .replace("}", "}}")
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
