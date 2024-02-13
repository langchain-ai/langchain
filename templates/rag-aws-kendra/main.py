from rag_aws_kendra.chain import chain

if __name__ == "__main__":
    query = "Does Kendra support table extraction?"

    print(chain.invoke(query))  # noqa: T201
