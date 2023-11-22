import json
from datetime import date
from typing import Iterable
from urllib.parse import urljoin

import pytest
import responses

from langchain.agents import create_cube_agent
from langchain.agents.agent_toolkits import CubeToolkit
from langchain.pydantic_v1 import SecretStr
from langchain.utilities.cube import CubeAPIWrapper, Query
from tests.unit_tests.llms.fake_llm import FakeLLM

CUBE_API_URL = "http://cube-api:4000"
CUBE_API_TOKEN = "TOKEN"
CUBES = [
    {
        "name": "Users",
        "title": "Users",
        "connectedComponent": 1,
        "measures": [
            {
                "name": "users.count",
                "title": "Users Count",
                "shortTitle": "Count",
                "aliasName": "users.count",
                "type": "number",
                "aggType": "count",
                "drillMembers": ["users.id", "users.city", "users.createdAt"],
            }
        ],
        "dimensions": [
            {
                "name": "users.city",
                "title": "Users City",
                "type": "string",
                "aliasName": "users.city",
                "shortTitle": "City",
                "suggestFilterValues": True,
            }
        ],
        "segments": [],
    },
    {
        "name": "Orders",
        "title": "Orders",
        "connectedComponent": 1,
        "measures": [
            {
                "name": "orders.count",
                "title": "Orders Count",
                "shortTitle": "Count",
                "aliasName": "orders.count",
                "type": "number",
                "aggType": "count",
                "drillMembers": ["orders.id", "orders.type", "orders.createdAt"],
            }
        ],
        "dimensions": [
            {
                "name": "orders.type",
                "title": "Orders Type",
                "type": "string",
                "aliasName": "orders.city",
                "shortTitle": "Type",
                "suggestFilterValues": True,
            }
        ],
        "segments": [],
    },
]


def _get_format_instructions() -> str:
    schema = Query.schema()

    # Remove extraneous fields.
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]

    return json.dumps(reduced_schema)


PROMPT = f"""You are an agent designed to interact with a Cube Semantic Layer.
Given an input question, create a syntactically correct Cube query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most relevant examples in the database.
Never query for all the columns from a specific model, only ask for the relevant columns given the question.
You have access to tools for interacting with the Cube Semantic Layer.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
If the question does not seem related to the Cube Semantic Layer, just return "I don't know" as the answer.


load_cube: Input to this tool is a detailed and correct Cube query, it format is JSON. Output is a result from the Cube, it format is JSON.This current date is {date.today().isoformat()}.If the query is not correct, an error message will be returned.If an error is returned, rewrite the query, check the query, and try again.
The input should be formatted as a JSON instance that conforms to the JSON schema below.
As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.
Here is the input schema:
```
{_get_format_instructions()}
```
meta_information_cube: Input to this tool is a comma-separated list of models, output is a Markdown table of the meta-information for those models.Be sure that the models actually exist by calling list_models_cube first!Example Input: "model1, model2, model3"
list_models_cube: Input is an empty string, output is a Markdown table of models in the Cube.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [load_cube, meta_information_cube, list_models_cube]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: I\'d like to get the data on our daily, weekly, and monthly active users.
Thought: I should look at the models in the Cube to see what I can query.  Then I should query the meta-information of the most relevant models.
"""  # noqa: E501
LIST_MODELS_CUBE_PROMPT = (
    PROMPT
    + """Action: list_models_cube
Action Input: ""
Observation: | Model | Description |
| --- | --- |
| Users | Users |
| Orders | Orders |

Thought:"""
)
META_INFORMATION_CUBE_PROMPT = (
    LIST_MODELS_CUBE_PROMPT
    + """Action: meta_information_cube
Action Input: "Users"
Observation: ## Model: Users

### Measures:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Count |  | users.count | number |

### Dimensions:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| City |  | users.city | string |

Thought:"""
)
LOAD_CUBE_PROMPT = (
    META_INFORMATION_CUBE_PROMPT
    + """Action: load_cube
Action Input: "{"measures":["stories.count"],"dimensions":["stories.category"],"filters":[{"member":"stories.isDraft","operator":"equals","values":["No"]}],"timeDimensions":[{"dimension":"stories.time","dateRange":["2015-01-01","2015-12-31"],"granularity":"month"}],"limit":100,"offset":50,"order":{"stories.time":"asc","stories.count":"desc"}}"
Observation: [{"users.count": "700"}]
Thought:"""  # noqa: E501
)
QUERIES = {
    PROMPT: 'Action: list_models_cube\nAction Input: ""',
    LIST_MODELS_CUBE_PROMPT: 'Action: meta_information_cube\nAction Input: "Users"',
    META_INFORMATION_CUBE_PROMPT: "Action: load_cube\nAction Input: "
    '"{"measures":["stories.count"],"dimensions":["stories.category"],"filters":[{"member":"stories.isDraft","operator":"equals","values":["No"]}],"timeDimensions":[{"dimension":"stories.time","dateRange":["2015-01-01","2015-12-31"],"granularity":"month"}],"limit":100,"offset":50,"order":{"stories.time":"asc","stories.count":"desc"}}"',
    LOAD_CUBE_PROMPT: "Final Answer: The daily, weekly, and monthly active users are"
    " all 700.",
}


@pytest.fixture(autouse=True)
def mocked_responses() -> Iterable[responses.RequestsMock]:
    """Fixture mocking requests.get."""
    with responses.RequestsMock() as rsps:
        yield rsps


def mocked_mata(mocked_responses: responses.RequestsMock) -> None:
    mocked_responses.add(
        method=responses.GET,
        url=urljoin(CUBE_API_URL, "/meta"),
        body=json.dumps({"cubes": CUBES}),
    )


def mocked_load(mocked_responses: responses.RequestsMock) -> None:
    mocked_responses.add(
        method=responses.POST,
        url=urljoin(CUBE_API_URL, "/load"),
        body=json.dumps(
            {
                "query": {
                    "measures": ["users.count"],
                    "filters": [],
                    "timezone": "UTC",
                    "dimensions": [],
                    "timeDimensions": [],
                },
                "data": [{"users.count": "700"}],
                "annotation": {
                    "measures": {
                        "users.count": {
                            "title": "Users Count",
                            "shortTitle": "Count",
                            "type": "number",
                        }
                    },
                    "dimensions": {},
                    "segments": {},
                    "timeDimensions": {},
                },
            }
        ),
    )


def test_create_cube_agent(mocked_responses: responses.RequestsMock) -> None:
    mocked_load(mocked_responses)

    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
    )

    llm = FakeLLM(queries=QUERIES)

    toolkit = CubeToolkit(
        cube=cube,
        llm=llm,
    )

    agent_executor = create_cube_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
    )

    question = (
        "I'd like to get the data on our daily, weekly, and monthly active users."
    )
    answer = "The daily, weekly, and monthly active users are all 700."

    assert agent_executor.run(question) == answer
