# flake8: noqa
REQUEST_TEMPLATE = """You are a helpful AI Assistant. Please provide JSON arguments to agentFunc() based on the user's instructions.

API_SCHEMA: ```typescript
{schema}
```

USER_INSTRUCTIONS: "{instructions}"

Your arguments must be plain json provided in a markdown block:

ARGS: ```json
{{valid json conforming to API_SCHEMA}}
```

Example
-----

ARGS: ```json
{{"foo": "bar", "baz": {{"qux": "quux"}}}}
```

The block must be no more than 1 line long, and all arguments must be valid JSON. All string arguments must be wrapped in double quotes.
You MUST strictly comply to the types indicated by the provided schema, including all required args.

If you don't have sufficient information to call the function due to things like requiring specific uuid's, you can reply with the following message:

Message: ```text
Concise response requesting the additional information that would make calling the function successful.
```

Begin
-----
ARGS:
"""
RESPONSE_TEMPLATE = """You are a helpful AI assistant trained to answer user queries from API responses.
You attempted to call an API, which resulted in:
API_RESPONSE: {response}

USER_COMMENT: "{instructions}"


If the API_RESPONSE can answer the USER_COMMENT respond with the following markdown json block:
Response: ```json
{{"response": "Concise response to USER_COMMENT based on API_RESPONSE."}}
```

Otherwise respond with the following markdown json block:
Response Error: ```json
{{"response": "What you did and a concise statement of the resulting error. If it can be easily fixed, provide a suggestion."}}
```

You MUST respond as a markdown json code block.

Begin:
---
"""
