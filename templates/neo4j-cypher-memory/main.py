from neo4j_cypher_memory.chain import chain

if __name__ == "__main__":
    original_query = "Who played in Top Gun?"
    print(  # noqa: T201
        chain.invoke(
            {
                "question": original_query,
                "user_id": "user_123",
                "session_id": "session_1",
            }
        )
    )
    follow_up_query = "Did they play in any other movies?"
    print(  # noqa: T201
        chain.invoke(
            {
                "question": follow_up_query,
                "user_id": "user_123",
                "session_id": "session_1",
            }
        )
    )
