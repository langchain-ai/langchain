import json
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from sql_research_assistant.search.sql import sql_answer_chain

RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()


def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)  # noqa: T201
        return f"Failed to retrieve the webpage: {e}"


def web_search(query: str, num_results: int):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{agent_prompt}"),
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

AUTO_AGENT_INSTRUCTIONS = """
This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific agent, defined by its type and role, with each agent requiring distinct instructions.
Agent
The agent is determined by the field of the topic and the specific name of the agent that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each agent type is associated with a corresponding emoji.

examples:
task: "should I invest in apple stocks?"
response: 
{
    "agent": "💰 Finance Agent",
    "agent_role_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
}
task: "could reselling sneakers become profitable?"
response: 
{ 
    "agent":  "📈 Business Analyst Agent",
    "agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
}
task: "what are the most interesting sites in Tel Aviv?"
response:
{
    "agent:  "🌍 Travel Agent",
    "agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
}
"""  # noqa: E501
CHOOSE_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=AUTO_AGENT_INSTRUCTIONS), ("user", "task: {task}")]
)

SUMMARY_TEMPLATE = """{text} 

-----------

Using the above text, answer in short the following question: 

> {question}
 
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

scrape_and_summarize: Runnable[Any, Any] = (
    RunnableParallel(
        {
            "question": lambda x: x["question"],
            "text": lambda x: scrape_text(x["url"])[:10000],
            "url": lambda x: x["url"],
        }
    )
    | RunnableParallel(
        {
            "summary": SUMMARY_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser(),
            "url": lambda x: x["url"],
        }
    )
    | RunnableLambda(lambda x: f"Source Url: {x['url']}\nSummary: {x['summary']}")
)


def load_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


search_query = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | load_json
choose_agent = (
    CHOOSE_AGENT_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | load_json
)

get_search_queries = (
    RunnablePassthrough().assign(
        agent_prompt=RunnableParallel({"task": lambda x: x})
        | choose_agent
        | (lambda x: x.get("agent_role_prompt"))
    )
    | search_query
)


chain = (
    get_search_queries
    | (lambda x: [{"question": q} for q in x])
    | sql_answer_chain.map()
    | (lambda x: "\n\n".join(x))
)
