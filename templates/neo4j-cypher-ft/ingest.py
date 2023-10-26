from langchain.graphs import Neo4jGraph

graph = Neo4jGraph()

# Import sample data 
graph.query(
    """
MERGE (m:Movie {name:"Top Gun"})
WITH m
UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
MERGE (a:Person {name:actor})
MERGE (a)-[:ACTED_IN]->(m)
"""
)

# Create full text index for entity matching
# on Person and Movie nodes
graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS"
    " FOR (m:Movie|Person) ON EACH [m.title, m.name]"
)
