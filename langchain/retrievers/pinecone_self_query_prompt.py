# flake8: noqa
from typing import Dict

from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseOutputParser

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
QUESTION: What was the sentiment of Mexican and Taiwanese media outlets regarding the 2024 trade deal between America and India?
DOCUMENT STORE QUERY:
```json
{{
    "search_string": "Trade deal between America and India",
    "metadata_filter": {{
        "country": {{
            "$in": ["Mexico", "United Mexican States", "mexico", "Taiwan", "Republic of China", "ROC", "taiwan"]
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
