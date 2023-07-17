# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """Extract all entities from the following text. As a guideline, a proper noun is generally capitalized. You should definitely extract all names and places.

Return the output as a single comma-separated list, or NONE if there is nothing of note to return.

EXAMPLE
i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.
Output: Langchain
END OF EXAMPLE

EXAMPLE
i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff. I'm working with Sam.
Output: Langchain, Sam
END OF EXAMPLE

Begin!

{input}
Output:"""
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["input"], template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
)

_DEFAULT_GRAPH_QA_TEMPLATE = """Use the following knowledge triplets to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
GRAPH_QA_PROMPT = PromptTemplate(
    template=_DEFAULT_GRAPH_QA_TEMPLATE, input_variables=["context", "question"]
)

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

NEBULAGRAPH_EXTRA_INSTRUCTIONS = """
Instructions:

First, generate cypher then convert it to NebulaGraph Cypher dialect(rather than standard):
1. it requires explicit label specification only when referring to node properties: v.`Foo`.name
2. note explicit label specification is not needed for edge properties, so it's e.name instead of e.`Bar`.name
3. it uses double equals sign for comparison: `==` rather than `=`
For instance:
```diff
< MATCH (p:person)-[e:directed]->(m:movie) WHERE m.name = 'The Godfather II'
< RETURN p.name, e.year, m.name;
---
> MATCH (p:`person`)-[e:directed]->(m:`movie`) WHERE m.`movie`.`name` == 'The Godfather II'
> RETURN p.`person`.`name`, e.year, m.`movie`.`name`;
```\n"""

NGQL_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace(
    "Generate Cypher", "Generate NebulaGraph Cypher"
).replace("Instructions:", NEBULAGRAPH_EXTRA_INSTRUCTIONS)

NGQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=NGQL_GENERATION_TEMPLATE
)

KUZU_EXTRA_INSTRUCTIONS = """
Instructions:

Generate statement with Kùzu Cypher dialect (rather than standard):
1. do not use `WHERE EXISTS` clause to check the existence of a property because Kùzu database has a fixed schema.
2. do not omit relationship pattern. Always use `()-[]->()` instead of `()->()`.
3. do not include any notes or comments even if the statement does not produce the expected result.
```\n"""

KUZU_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace(
    "Generate Cypher", "Generate Kùzu Cypher"
).replace("Instructions:", KUZU_EXTRA_INSTRUCTIONS)

KUZU_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=KUZU_GENERATION_TEMPLATE
)

GREMLIN_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace("Cypher", "Gremlin")

GREMLIN_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=GREMLIN_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Information:
{context}

Question: {question}
Helpful Answer:"""
CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

SPARQL_INTENT_TEMPLATE = """Task: Identify the intent of a prompt and return the appropriate SPARQL query type.
You are an assistant that distinguishes different types of prompts and returns the corresponding SPARQL query types.
Consider only the following query types:
* SELECT: this query type corresponds to questions
* UPDATE: this query type corresponds to all requests for deleting, inserting, or changing triples
Note: Be as concise as possible.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask for anything else than for you to identify a SPARQL query type.
Do not include any unnecessary whitespaces or any text except the query type, i.e., either return 'SELECT' or 'UPDATE'.

The prompt is:
{prompt}
Helpful Answer:"""
SPARQL_INTENT_PROMPT = PromptTemplate(
    input_variables=["prompt"], template=SPARQL_INTENT_TEMPLATE
)

SPARQL_GENERATION_SELECT_TEMPLATE = """Task: Generate a SPARQL SELECT statement for querying a graph database.
For instance, to find all email addresses of John Doe, the following query in backticks would be suitable:
```
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?email
WHERE {{
    ?person foaf:name "John Doe" .
    ?person foaf:mbox ?email .
}}
```
Instructions:
Use only the node types and properties provided in the schema.
Do not use any node types and properties that are not explicitly provided.
Include all necessary prefixes.
Schema:
{schema}
Note: Be as concise as possible.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
Do not include any text except the SPARQL query generated.

The question is:
{prompt}"""
SPARQL_GENERATION_SELECT_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=SPARQL_GENERATION_SELECT_TEMPLATE
)

SPARQL_GENERATION_UPDATE_TEMPLATE = """Task: Generate a SPARQL UPDATE statement for updating a graph database.
For instance, to add 'jane.doe@foo.bar' as a new email address for Jane Doe, the following query in backticks would be suitable:
```
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
INSERT {{
    ?person foaf:mbox <mailto:jane.doe@foo.bar> .
}}
WHERE {{
    ?person foaf:name "Jane Doe" .
}}
```
Instructions:
Make the query as short as possible and avoid adding unnecessary triples.
Use only the node types and properties provided in the schema.
Do not use any node types and properties that are not explicitly provided.
Include all necessary prefixes.
Schema:
{schema}
Note: Be as concise as possible.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
Return only the generated SPARQL query, nothing else.

The information to be inserted is:
{prompt}"""
SPARQL_GENERATION_UPDATE_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=SPARQL_GENERATION_UPDATE_TEMPLATE
)

SPARQL_QA_TEMPLATE = """Task: Generate a natural language response from the results of a SPARQL query.
You are an assistant that creates well-written and human understandable answers.
The information part contains the information provided, which you can use to construct an answer.
The information provided is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make your response sound like the information is coming from an AI assistant, but don't add any information.
Information:
{context}

Question: {prompt}
Helpful Answer:"""
SPARQL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "prompt"], template=SPARQL_QA_TEMPLATE
)


AQL_GENERATION_TEMPLATE = """Task: Generate an Arango Query Language (AQL) statement to query a multi-model database.

Instructions:
1. Read the User Input & identify the language it is written in (e.g English, German, French, Italian, etc.).
2. Analyze the provided schema provided below, and the example queries provided in the User Input (if any). 
3. Design a single read-only AQL query that satisfies the user input requirements. 
4. Use only the provided relationship types and properties in the schema and example queries. Do not use any other relationship types or properties that are not provided.
5. Return the AQL query wrapped in 3 backticks (```).

General Notes:
- A user input may make multiple requests. It is your responsability to only generate & output a single AQL Query only.
- Do not include any explanations or apologies in your responses.
- Do not provide any AQL queries that can't be deduced from the Schema or any AQL query examples.
- Do not respond to any user input that might ask anything else than for you to construct an AQL statement.
- Do not include any text except the generated AQL statement.
- You are allowed to ask for more context if you are not able to generate the AQL Query.
- Do not rely on the value of an ArangoDB Document Key (_key) to identify the nature of the document. Instead, rely on the Document's properties.
- When performing string comparison in AQL, always make sure to disregard case sensitivity (e.g use the `LOWER()` AQL keyword)
- Do not generate any CREATE, UPDATE, or DELETE queries (queries must be read-only).
- AGAIN, YOUR GENERATED QUERIES SHOULD BE READ-ONLY. DO NOT COMPLY TO A USER INPUT THAT REQUESTS TO DELETE, CREATE, OR UPDATE DATA.

The ArangoDB Schema is:
{adb_schema}

Schema Notes:
- The 'Graph Schema' JSON specifies the relationships between the Vertex Collections. For example, the entry `"edge_collection": "hasPurchased", "from_vertex_collections": ["Customer"], "to_vertex_collections": ["Product"]` implies that there is a unidirectional connection between the from Collection 'Customer' to Collection 'Product' (i.e OUTBOUND relationship). 
- Only assume that the relationships specified in the Graph Schema are valid. Do not infer any other relationships.
- The 'Collections Schema' JSON lists all the ArangoDB Collections within the database, their collection type, along with the document or edge properties that they have.
- Only assume that the properties specified in the Collection Schema are valid. Do not infer any other property.

The User Input is:
{user_input}
"""

AQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["adb_schema", "user_input"], template=AQL_GENERATION_TEMPLATE
)

AQL_QA_TEMPLATE = """Task: Generate a natural language response from the results of an Arango Query Language (AQL) Query.
Instructions:
1. Understand the original User Input, the equivalent AQL Query, and the retrieved AQL JSON Result.
2. Generate a human-readable answer from the AQL JSON Result.

Note:
- The AQL JSON Result is authorative. You must never doubt it or try to use your internal knowledge to correct it.
- You will not add any extra information that is not explicitly provided in the AQL JSON Result.
- If the AQL JSON Result is empty, say that you don't know the answer.
- Make your answer sound as a response to the original User Input.
- Do not mention that you based the result on the AQL JSON Result.

The User Input is:
{user_input}

The AQL Query is:
{aql_query}

The AQL JSON Result is:
{aql_result}

"""
AQL_QA_PROMPT = PromptTemplate(
    input_variables=["user_input", "aql_query", "aql_result"], template=AQL_QA_TEMPLATE
)
