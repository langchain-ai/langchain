from typing import Any, Dict, List, Union

from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain.memory import ChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Connection to Neo4j
graph = Neo4jGraph()

# Cypher validation tool for relationship directions
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph.structured_schema.get("relationships")
]
cypher_validation = CypherQueryCorrector(corrector_schema)

# LLMs
cypher_llm = ChatOpenAI(model="gpt-4", temperature=0.0)
qa_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)


def convert_messages(input: List[Dict[str, Any]]) -> ChatMessageHistory:
    history = ChatMessageHistory()
    for item in input:
        history.add_user_message(item["result"]["question"])
        history.add_ai_message(item["result"]["answer"])
    return history


def get_history(input: Dict[str, Any]) -> ChatMessageHistory:
    input.pop("question")
    # Lookback conversation window
    window = 3
    data = graph.query(
        """
    MATCH (u:User {id:$user_id})-[:HAS_SESSION]->(s:Session {id:$session_id}),
                       (s)-[:LAST_MESSAGE]->(last_message)
    MATCH p=(last_message)<-[:NEXT*0.."""
        + str(window)
        + """]-()
    WITH p, length(p) AS length
    ORDER BY length DESC LIMIT 1
    UNWIND reverse(nodes(p)) AS node
    MATCH (node)-[:HAS_ANSWER]->(answer)
    RETURN {question:node.text, answer:answer.text} AS result
 """,
        params=input,
    )
    history = convert_messages(data)
    return history.messages


def save_history(input):
    print(input)
    if input.get("function_response"):
        input.pop("function_response")
    # store history to database
    graph.query(
        """MERGE (u:User {id: $user_id})
WITH u                
OPTIONAL MATCH (u)-[:HAS_SESSION]->(s:Session{id: $session_id}),
                (s)-[l:LAST_MESSAGE]->(last_message)
FOREACH (_ IN CASE WHEN last_message IS NULL THEN [1] ELSE [] END |
CREATE (u)-[:HAS_SESSION]->(s1:Session {id:$session_id}),
    (s1)-[:LAST_MESSAGE]->(q:Question {text:$question, cypher:$query, date:datetime()}),
        (q)-[:HAS_ANSWER]->(:Answer {text:$output}))                                
FOREACH (_ IN CASE WHEN last_message IS NOT NULL THEN [1] ELSE [] END |
CREATE (last_message)-[:NEXT]->(q:Question 
                {text:$question, cypher:$query, date:datetime()}),
                (q)-[:HAS_ANSWER]->(:Answer {text:$output}),
                (s)-[:LAST_MESSAGE]->(q)
DELETE l)                """,
        params=input,
    )

    # Return LLM response to the chain
    return input["output"]


# Generate Cypher statement based on natural language input
cypher_template = """This is important for my career.
Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question:
{schema}

Question: {question}
Cypher query:"""  # noqa: E501

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question, convert it to a Cypher query. No pre-amble.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", cypher_template),
    ]
)

cypher_response = (
    RunnablePassthrough.assign(schema=lambda _: graph.get_schema, history=get_history)
    | cypher_prompt
    | cypher_llm.bind(stop=["\nCypherResult:"])
    | StrOutputParser()
)

# Generate natural language response based on database results
response_system = """You are an assistant that helps to form nice and human 
understandable answers based on the provided information from tools.
Do not add any other information that wasn't present in the tools, and use 
very concise style in interpreting results!
"""

response_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=response_system),
        HumanMessagePromptTemplate.from_template("{question}"),
        MessagesPlaceholder(variable_name="function_response"),
    ]
)


def get_function_response(
    query: str, question: str
) -> List[Union[AIMessage, ToolMessage]]:
    context = graph.query(cypher_validation(query))
    TOOL_ID = "call_H7fABDuzEau48T10Qn0Lsh0D"
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": TOOL_ID,
                        "function": {
                            "arguments": '{"question":"' + question + '"}',
                            "name": "GetInformation",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(content=str(context), tool_call_id=TOOL_ID),
    ]
    return messages


chain = (
    RunnablePassthrough.assign(query=cypher_response)
    | RunnablePassthrough.assign(
        function_response=lambda x: get_function_response(x["query"], x["question"]),
    )
    | RunnablePassthrough.assign(
        output=response_prompt | qa_llm | StrOutputParser(),
    )
    | save_history
)

# Add typing for input


class Question(BaseModel):
    question: str
    user_id: str
    session_id: str


chain = chain.with_types(input_type=Question)
