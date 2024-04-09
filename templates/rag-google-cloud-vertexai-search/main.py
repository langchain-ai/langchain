from rag_google_cloud_vertexai_search.chain import chain

if __name__ == "__main__":
    query = "Who is the CEO of Google Cloud?"
    print(chain.invoke(query))  # noqa: T201
