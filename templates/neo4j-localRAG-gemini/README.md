# RAG using Gemini for local Neo4j databases
This template aims to streamline the processes 
1. Generating cypher query for the asked question
2. Executing the generate cypher query on the local Neo4j database
3. Displaying results for the asked question

# Environment setup
Define the following environment variables
```
Neo4j_url = 'neo4j://localhost:7687' # Default URL for local Neo4j databases, change if needed.
Neo4j_password = YOUR-NEO4J-DATABASE-PASSWORD
Gemini_api_key = YOUR-GEMINI-API-KEY
```
