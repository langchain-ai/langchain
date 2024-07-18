API_CONTROLLER_PROMPT = (
    "You are turning user input into a json query"
    """ for an API request tool.

The final output to the tool should be a json string with a single key "data".
The value of "data" should be a dictionary of key-value pairs you want """
    """to POST to the url.
Always use double quotes for strings in the json string.
Always respond only with the json object and nothing else.

Here is documentation on the API:
Base url: {api_url}
Endpoint documentation:
{api_docs}

User Input: {input}
"""
)
