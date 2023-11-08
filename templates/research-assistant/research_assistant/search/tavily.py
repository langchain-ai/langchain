import os

from langchain.schema.runnable import RunnableLambda
from tavily import Client

client = Client(os.environ["TAVILY_API_KEY"])
chain = RunnableLambda(client.advanced_search) | {"results": lambda x: x["results"]}
