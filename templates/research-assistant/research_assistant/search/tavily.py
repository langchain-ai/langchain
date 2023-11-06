from tavily import Client
import os
from langchain.schema.runnable import RunnableLambda

client = Client(os.environ["TAVILY_API_KEY"])
chain = RunnableLambda(client.advanced_search) | {"results": lambda x: x["results"]}
