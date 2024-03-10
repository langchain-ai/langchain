from astradb_entomology_rag import chain

if __name__ == "__main__":
    response = chain.invoke("Are there more coleoptera or bugs?")
    print(response)  # noqa: T201
