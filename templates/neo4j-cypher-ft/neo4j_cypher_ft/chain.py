from typing import List, Optional

from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

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


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)


# Fulltext index query
def map_to_database(entities: Entities) -> Optional[str]:
    result = ""
    for entity in entities.names:
        response = graph.query(
            "CALL db.index.fulltext.queryNodes('entity', $entity + '*', {limit:1})"
            " YIELD node,score RETURN node.name AS result",
            {"entity": entity},
        )
        try:
            result += f"{entity} maps to {response[0]['result']} in database\n"
        except IndexError:
            pass
    return result


entity_chain = create_structured_output_chain(Entities, qa_llm, prompt)

# Generate Cypher statement based on natural language input
cypher_template = """Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question:
{schema}
Entities in the question map to the following database values:
{entities_list}
Question: {question}
Cypher query:"""  # noqa: E501

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question, convert it to a Cypher query. No pre-amble.",
        ),
        ("human", cypher_template),
    ]
)

cypher_response = (
    RunnablePassthrough.assign(names=entity_chain)
    | RunnablePassthrough.assign(
        entities_list=lambda x: map_to_database(x["names"]["function"]),
        schema=lambda _: graph.get_schema,
    )
    | cypher_prompt
    | cypher_llm.bind(stop=["\nCypherResult:"])
    | StrOutputParser()
)

# Generate natural language response based on database results
response_template = """Based on the the question, Cypher query, and Cypher response, write a natural language response:
Question: {question}
Cypher query: {query}
Cypher Response: {response}"""  # noqa: E501

response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and Cypher response, convert it to a natural"
            " language answer. No pre-amble.",
        ),
        ("human", response_template),
    ]
)

chain = (
    RunnablePassthrough.assign(query=cypher_response)
    | RunnablePassthrough.assign(
        response=lambda x: graph.query(cypher_validation(x["query"])),
    )
    | response_prompt
    | qa_llm
    | StrOutputParser()
)

# Add typing for input


class Question(BaseModel):
    question: str


chain = chain.with_types(input_type=Question)
