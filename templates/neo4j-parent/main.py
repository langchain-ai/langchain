from neo4j_parent.chain import chain

if __name__ == "__main__":
    original_query = "What is the plot of the Dune?"
    print(chain.invoke(original_query))
