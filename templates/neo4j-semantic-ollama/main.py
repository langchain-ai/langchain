from neo4j_semantic_ollama import agent_executor

if __name__ == "__main__":
    original_query = "What do you know about person John?"
    followup_query = "John Travolta"
    chat_history = [
        (
            "What do you know about person John?",
            "I found multiple people named John. Could you please specify "
            "which one you are interested in? Here are some options:"
            "\n\n1. John Travolta\n2. John McDonough",
        )
    ]
    print(agent_executor.invoke({"input": original_query}))
    print(
        agent_executor.invoke({"input": followup_query, "chat_history": chat_history})
    )
