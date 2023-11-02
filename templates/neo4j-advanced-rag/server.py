from neo4j_advanced_rag import chain as neo4j_advanced_chain

add_routes(
    app, neo4j_advanced_chain, path="/neo4j-advanced-rag", config_keys=["configurable"]
)
