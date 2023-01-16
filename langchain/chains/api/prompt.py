# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

API_URL_PROMPT_TEMPLATE = """You are given the below API Documentation:
{api_docs}
Using this documentation, generate the full API url to call for answering the user question.
You should build the API url so as to get a response that is as short as possible, while still getting the necessary information to answer the question. Pay attention in the following example, how there are many pieces of information that are deliberately not included in the API call. You should do the same.

EXAMPLE
Question: Please tell me what the weather is like right now in London?
API url:https://api.open-meteo.com/v1/forecast?latitude=51.5074&longitude=0.1278&current_weather=true&timezone=auto
END OF EXAMPLE

Question:{question}
API url:
"""

API_URL_PROMPT = PromptTemplate(
    input_variables=[
        "api_docs",
        "question",
    ],
    template=API_URL_PROMPT_TEMPLATE,
)

API_RESPONSE_PROMPT_TEMPLATE = (
    API_URL_PROMPT_TEMPLATE
    + """ {api_url}

Here is the response from the API:

{api_response}

Summarize this response to answer the original question.

Summary:"""
)

API_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["api_docs", "question", "api_url", "api_response"],
    template=API_RESPONSE_PROMPT_TEMPLATE,
)
