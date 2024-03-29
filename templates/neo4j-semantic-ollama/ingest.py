from langchain_community.graphs import Neo4jGraph

# Instantiate connection to Neo4j
graph = Neo4jGraph()

# Define unique constraints
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE;")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;")

# Import movie information

movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies.csv'
AS row
CALL {
    WITH row
    MERGE (m:Movie {id:row.movieId})
    SET m.released = date(row.released),
        m.title = row.title,
        m.imdbRating = toFloat(row.imdbRating)
    FOREACH (director in split(row.director, '|') | 
        MERGE (p:Person {name:trim(director)})
        MERGE (p)-[:DIRECTED]->(m))
    FOREACH (actor in split(row.actors, '|') | 
        MERGE (p:Person {name:trim(actor)})
        MERGE (p)-[:ACTED_IN]->(m))
    FOREACH (genre in split(row.genres, '|') | 
        MERGE (g:Genre {name:trim(genre)})
        MERGE (m)-[:IN_GENRE]->(g))
} IN TRANSACTIONS
"""

graph.query(movies_query)

# Import rating information
rating_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/ratings.csv'
AS row
CALL {
    WITH row
    MATCH (m:Movie {id:row.movieId})
    MERGE (u:User {id:row.userId})
    MERGE (u)-[r:RATED]->(m)
    SET r.rating = toFloat(row.rating),
        r.timestamp = row.timestamp
} IN TRANSACTIONS OF 10000 ROWS
"""

graph.query(rating_query)

# Define fulltext indices
graph.query("CREATE FULLTEXT INDEX movie IF NOT EXISTS FOR (m:Movie) ON EACH [m.title]")
graph.query(
    "CREATE FULLTEXT INDEX person IF NOT EXISTS FOR (p:Person) ON EACH [p.name]"
)
