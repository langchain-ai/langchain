from neo4j_advanced_rag.chain import chain

if __name__ == "__main__":
    original_query = "What is the plot of the Dune?"
    print(  # noqa: T201
        chain.invoke(
            {"question": original_query},
            {"configurable": {"strategy": "parent_strategy"}},
        )
    )
