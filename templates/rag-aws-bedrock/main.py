from rag_aws_bedrock.chain import chain

if __name__ == "__main__":
    query = "What is this data about?"

    print(chain.invoke(query))  # noqa: T201
