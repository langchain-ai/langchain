from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()

graph.query(
    """
MERGE (m:Movie {name:"Top Gun"})
WITH m
UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
MERGE (a:Actor {name:actor})
MERGE (a)-[:ACTED_IN]->(m)
WITH a
WHERE a.name = "Tom Cruise"
MERGE (a)-[:ACTED_IN]->(:Movie {name:"Mission Impossible"})
"""
)
