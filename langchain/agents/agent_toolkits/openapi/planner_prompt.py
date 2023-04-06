# flake8: noqa

from langchain.prompts.prompt import PromptTemplate


API_PLANNER_PROMPT = """You are a planner that plans a sequence of API calls to assist with user queries against an API.

You should:
1) evaluate whether the user query can be solved by the API documentated below. If no, say why.
2) if yes, generate a plan of API calls and say what they are doing step by step.

You should only use API endpoints documented below ("Endpoints you can use:").
Some user queries can be resolved in a single API call, but some will require several API calls.
The plan will be passed to an API controller that can format it into web requests and return the responses.

----

Here are some examples:

Fake endpoints for examples:
GET /user to get information about the current user
GET /products/search search across products
POST /users/{{id}}/cart to add products to a user's cart

User query: tell me a joke
Plan: Sorry, this API's domain is shopping, not comedy.

Usery query: I want to buy a couch
Plan: 1. GET /products/search to search for couches
2. GET /user to find the user's id
3. POST /users/{{id}}/cart to add a couch to the user's cart

----

Here are endpoints you can use. Do not reference any of the endpoints above.

{endpoints}

----

User query: {query}
Plan:"""
API_PLANNER_TOOL_NAME = "api_planner"
API_PLANNER_TOOL_DESCRIPTION = f"Can be used to generate the right API calls to assist with a user query, like {API_PLANNER_TOOL_NAME}(query). Should always be called before trying to calling the API controller."

# Execution.
API_CONTROLLER_PROMPT = """You are an agent that gets a sequence of API calls and given their documentation, should execute them and return the final response.
If you cannot complete them and run into issues, you should explain the issue. If you're able to resolve an API call, you can retry the API call. When interacting with API objects, you should extract ids for inputs to other API calls but ids and names for outputs returned to the User.


Here is documentation on the API:
Base url: {api_url}
Endpoints:
{api_docs}


Here are tools to execute requests against the API: {tool_descriptions}


Starting below, you should follow this format:

Plan: the plan of API calls to execute
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the output of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing the plan (or, I cannot finish executing the plan without knowing some other information.)
Final Answer: the final output from executing the plan or missing information I'd need to re-plan correctly.


Begin!

Plan: {input}
Thought:
{agent_scratchpad}
"""
API_CONTROLLER_TOOL_NAME = "api_controller"
API_CONTROLLER_TOOL_DESCRIPTION = f"Can be used to execute a plan of API calls, like {API_CONTROLLER_TOOL_NAME}(plan)."

# Orchestrate planning + execution.
# The goal is to have an agent at the top-level (e.g. so it can recover from errors and re-plan) while
# keeping planning (and specifically the planning prompt) simple.
API_ORCHESTRATOR_PROMPT = """You are an agent that assists with user queries against API, things like querying information or creating resources.
Some user queries can be resolved in a single API call though some require several API call.
You should always plan your API calls first, and then execute the plan second.
You should never return information without executing the api_controller tool.


Here are the tools to plan and execute API requests: {tool_descriptions}


Starting below, you should follow this format:

User query: the query a User wants help with related to the API
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: the final output from executing the plan


Example:
User query: can you add some trendy stuff to my shopping cart.
Thought: I should plan API calls first.
Action: api_planner
Action Input: I need to find the right API calls to add trendy items to the users shopping cart
Observation: 1) GET /items/trending to get trending item ids
2) GET /user to get user
3) POST /cart to post the trending items to the user's cart
Thought: I'm ready to execute the API calls.
Action: api_controller
Action Input: 1) GET /items/trending to get trending item ids
2) GET /user to get user
3) POST /cart to post the trending items to the user's cart
...

Begin!

User query: {input}
Thought: I should generate a plan to help with this query and then copy that plan exactly to the controller.
{agent_scratchpad}"""

REQUESTS_GET_TOOL_DESCRIPTION = """Use this to GET content from a website.
Input to the tool should be a json string with 2 keys: "url" and "output_instructions".
The value of "url" should be a string. The value of "output_instructions" should be instructions on what information to extract from the response, for example the id(s) for a resource(s) that the GET request fetches.
"""

PARSING_GET_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
When working with API objects, you should usually use ids over names.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response", "instructions"],
)

REQUESTS_POST_TOOL_DESCRIPTION = """Use this when you want to POST to a website.
Input to the tool should be a json string with 3 keys: "url", "data", and "output_instructions".
The value of "url" should be a string.
The value of "data" should be a dictionary of key-value pairs you want to POST to the url.
The value of "summary_instructions" should be instructions on what information to extract from the response, for example the id(s) for a resource(s) that the POST request creates.
Always use double quotes for strings in the json string."""

PARSING_POST_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
When working with API objects, you should usually use ids over names. Do not return any ids or names that are not in the response.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response", "instructions"],
)
