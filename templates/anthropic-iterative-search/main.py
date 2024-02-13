from anthropic_iterative_search import final_chain

if __name__ == "__main__":
    query = (
        "Which movie came out first: Oppenheimer, or "
        "Are You There God It's Me Margaret?"
    )
    print(  # noqa: T201
        final_chain.with_config(configurable={"chain": "retrieve"}).invoke(
            {"query": query}
        )
    )
