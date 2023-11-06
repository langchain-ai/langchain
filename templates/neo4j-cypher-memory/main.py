from neo4j_cypher_memory.chain import chain

if __name__ == "__main__":
    original_query = "Who played in Top Gun?"
    print(chain.invoke({"question": original_query, "user_id": "123"}))
    follow_up_query = "Did they play in any other movies?"
    print(chain.invoke({"question": follow_up_query, "user_id": "123"}))
