import json
from abc import abstractmethod
from typing import Any, Dict, List

from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import VectorStoreInfo
from langchain.schema import BaseOutputParser, BaseRetriever, Document
from langchain.vectorstores.base import VectorStoreRetriever

pinecone_format_instructions = """RESPONSE FORMAT
----------------------------

When responding to me please, please output a response in this format. Use a markdown code snippet formatted in the following schema:

```json
{{{{
    "search_string": string \\ The string to compare to document contents,
    "metadata_filter": {{{{
        "<MetadataField>": {{{{
            <Operator>: <Value>
        }}}},
        "<MetadataField>": {{{{
            <Operator>: <Value>
        }}}},
    }}}} \\ The conditions on which to filter the metadata
}}}}
```

Filtering metadata supports the following operators:
    $eq - Equal to (number, string, boolean)
    $ne - Not equal to (number, string, boolean)
    $gt - Greater than (number)
    $gte - Greater than or equal to (number)
    $lt - Less than (number)
    $lte - Less than or equal to (number)
    $in - In array (string or number)
    $nin - Not in array (string or number)"""

pinecone_example = """EXAMPLE
----------------------------

Here is an example:

DOCUMENT STORE DESCRIPTION: News headlines from around the world
METADATA FIELDS: {{{{
    "year": "The year the headline was published", 
    "country": The country of origin of the news media outlet, 
    "source": "The name of the news media outlet"
}}}}
QUESTION: What was the sentiment of Chinese and Taiwanese media outlets regarding the 2024 trade deal between America and India?
DOCUMENT STORE QUERY:
```json
{{{{
    "search_string": "Trade deal between America and India",
    "metadata_filter": {{{{
        "country": {{{{
            "$in": ["China", "Taiwan"]
        }}}},
        "year": {{{{
            "$eq": 2024
        }}}}
    }}}}
}}}}
```"""

self_query_prompt = """INSTRUCTIONS
----------------------------

You have access to a store of documents. Each document contains text and a key-value store of associated metadata. Given a user question, your job is to come up with a fully formed query to the document store that will return the most relevant documents.

A document store query consists of two components: a search string and a metadata filter. The search string is compared to the text contents of the stored documents. The metadata filter is used to filter out documents whose metadata does not match the given criteria.


{format_instructions}


{example}


Begin!

DOCUMENT STORE DESCRIPTION: {{docstore_description}}
METADATA FIELDS: {{metadata_fields}}
QUESTION: {{{{question}}}}
DOCUMENT STORE QUERY:""".format(
    format_instructions=pinecone_format_instructions, example=pinecone_example
)


class VectorStoreQueryOutputParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        return pinecone_format_instructions

    def parse(self, text: str) -> Any:
        cleaned_output = text.strip()
        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json") :]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```") :]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.replace("`", "")
        cleaned_output = cleaned_output.strip()
        response = json.loads(cleaned_output)
        return {
            "query": response["search_string"],
            "filter": response["metadata_filter"],
        }


class VectorStoreExtendedInfo(VectorStoreInfo):
    metadata_field_descriptions: Dict[str, str]


class VectorStoreSelfQueryRetriever(VectorStoreRetriever):
    llm_chain: LLMChain

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        inputs = self.llm_chain.prep_inputs(query)
        vecstore_query = self.llm_chain.predict_and_parse(**inputs)
        print(vecstore_query)
        _query = vecstore_query["query"]
        _filter = vecstore_query["filter"]
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                _query, filter=_filter, **self.search_kwargs
            )
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                _query, filter=_filter, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        pass

    @classmethod
    def from_vecstore_extended_info(cls, llm, vecstore_extended_info, **kwargs):
        metadata_field_json = (
            "{" + json.dumps(vecstore_extended_info.metadata_field_descriptions) + "}"
        )
        _prompt = self_query_prompt.format(
            docstore_description=vecstore_extended_info.description,
            metadata_fields=metadata_field_json,
        )
        prompt = PromptTemplate(
            input_variables=["question"],
            template=_prompt,
            output_parser=VectorStoreQueryOutputParser(),
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            llm_chain=llm_chain,
            vectorstore=vecstore_extended_info.vectorstore,
            **kwargs,
        )


class PerfectRetriever(BaseRetriever):
    agent_executor: AgentExecutor

    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """

    @abstractmethod
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        pass
