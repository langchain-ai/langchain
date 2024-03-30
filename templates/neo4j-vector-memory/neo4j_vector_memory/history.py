from typing import Any, Dict, List, Union

from langchain.memory import ChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_core.messages import AIMessage, HumanMessage

graph = Neo4jGraph()


def convert_messages(input: List[Dict[str, Any]]) -> ChatMessageHistory:
    history = ChatMessageHistory()
    for item in input:
        history.add_user_message(item["result"]["question"])
        history.add_ai_message(item["result"]["answer"])
    return history


def get_history(input: Dict[str, Any]) -> List[Union[HumanMessage, AIMessage]]:
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


def save_history(input: Dict[str, Any]) -> str:
    input["context"] = [el.metadata["id"] for el in input["context"]]
    has_history = bool(input.pop("chat_history"))

    # store history to database
    if has_history:
        graph.query(
            """
MATCH (u:User {id: $user_id})-[:HAS_SESSION]->(s:Session{id: $session_id}),
                    (s)-[l:LAST_MESSAGE]->(last_message)
CREATE (last_message)-[:NEXT]->(q:Question 
                {text:$question, rephrased:$rephrased_question, date:datetime()}),
                (q)-[:HAS_ANSWER]->(:Answer {text:$output}),
                (s)-[:LAST_MESSAGE]->(q)
DELETE l
WITH q
UNWIND $context AS c
MATCH (n) WHERE elementId(n) = c
MERGE (q)-[:RETRIEVED]->(n)
""",
            params=input,
        )

    else:
        graph.query(
            """MERGE (u:User {id: $user_id})
CREATE (u)-[:HAS_SESSION]->(s1:Session {id:$session_id}),
    (s1)-[:LAST_MESSAGE]->(q:Question 
            {text:$question, rephrased:$rephrased_question, date:datetime()}),
        (q)-[:HAS_ANSWER]->(:Answer {text:$output})
WITH q
UNWIND $context AS c
MATCH (n) WHERE elementId(n) = c
MERGE (q)-[:RETRIEVED]->(n)
""",
            params=input,
        )

    # Return LLM response to the chain
    return input["output"]
