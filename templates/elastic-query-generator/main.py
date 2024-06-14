from elastic_query_generator.chain import chain

if __name__ == "__main__":
    print(chain.invoke({"input": "how many customers named Carol"}))
