# flake8: noqa

from langchain.prompts.prompt import PromptTemplate

REQUEST_TOOL_DESCRIPTION = """Use this when you want to POST to an API.
Input to the tool should be a json string with 3 keys: "url", "data", and "output_instructions".
The value of "url" should be a string.
The value of "data" should be a dictionary of key-value pairs you want to POST to the url.
The value of "output_instructions" should be instructions on what information to extract from the response,
for example the id(s) for a resource(s) that the POST request creates.
Always use double quotes for strings in the json string."""

TOOLKIT_TOOL_DESCRIPTION = """{description}. The tool must be invoked with a complete sentence starting with "{name}" and additional information on {required_params}."""


API_CONTROLLER_PROMPT = """You are turning user input into a json query for an API request tool.

The final output to the tool should be a json string with 2 keys: "url", "data".
The value of "url" should be a string.
The value of "data" should be a dictionary of key-value pairs you want to POST to the url.
Always use double quotes for strings in the json string.
Always respond only with the json object and nothing else.

Here is documentation on the API:
Base url: {api_url}
Endpoint documentation:
{api_docs}

User Input: {input}
"""