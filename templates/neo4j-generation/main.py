from neo4j_generation.chain import chain

if __name__ == "__main__":
    text = "Harrison works at LangChain, which is located in San Francisco"
    allowed_nodes = ["Person", "Organization", "Location"]
    allowed_relationships = ["WORKS_AT", "LOCATED_IN"]
    print(  # noqa: T201
        chain(
            text,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
        )
    )
