from neo4j_cypher.chain import chain

if __name__ == "__main__":
    original_query = "Who played in Top Gun?"
    print(chain.invoke({"question": original_query}))  # noqa: T201
