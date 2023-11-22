# flake8: noqa

from langchain.prompts.prompt import PromptTemplate

REQUESTS_GET_TOOL_DESCRIPTION = """Use this to GET content from an API.
Input to the tool should be a json string with 3 keys: "url", "params" and "output_instructions".
The value of "url" should be a string.
The value of "params" should be a dict of the needed and available
parameters from the OpenAPI spec related to the endpoint.
The value of "output_instructions" should be instructions on what information to extract from the response,
for example the id(s) for a resource(s) that the GET request fetches.
"""

REQUESTS_POST_TOOL_DESCRIPTION = """Use this when you want to POST to an API.
Input to the tool should be a json string with 3 keys: "url", "data", and "output_instructions".
The value of "url" should be a string.
The value of "data" should be a dictionary of key-value pairs you want to POST to the url.
The value of "output_instructions" should be instructions on what information to extract from the response,
for example the id(s) for a resource(s) that the POST request creates.
Always use double quotes for strings in the json string."""

REQUESTS_RESPONSE_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response", "instructions"],
)

TOOLKIT_TOOL_DESCRIPTION = """{description}. The tool must be invoked with a complete sentence starting with "{name}" and additional information on {required_params}."""


API_CONTROLLER_PROMPT = """You are an agent designed to answer questions by making web requests
to an API given given the documentation of an API endpoint and should execute it and return the final response.
If you cannot complete them and run into issues, you should explain the issue. If you're unable to resolve an API call, you can retry the API call.
When interacting with API objects, you should extract ids for inputs to other API calls but ids and names for outputs returned to the User.

Here is documentation on the API:
Base url: {api_url}
Endpoint documentation:
{api_docs}


Here are tools to execute requests against the API: {tool_descriptions}


Starting below, you should follow this format:

Question: the question to be answered by making the API calls
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the output of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing the plan (or, I cannot finish executing the plan without knowing some other information.)
Final Answer: the final output from executing the plan or missing information I'd need to re-plan correctly.


Begin!

Question: {input}
Thought:
{agent_scratchpad}
"""
