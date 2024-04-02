from neo4j_vector_memory.chain import chain

if __name__ == "__main__":
    user_id = "user_id_1"
    session_id = "session_id_1"
    original_query = "What is the plot of the Dune?"
    print(  # noqa: T201
        chain.invoke(
            {"question": original_query, "user_id": user_id, "session_id": session_id}
        )
    )
    follow_up_query = "Tell me more about Leto"
    print(  # noqa: T201
        chain.invoke(
            {"question": follow_up_query, "user_id": user_id, "session_id": session_id}
        )
    )
