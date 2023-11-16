from langchain.retrievers import WikipediaRetriever
from langchain.tools import tool

# This is used to tell the model how to best use the retriever.

retriever_description = """You will be asked a question by a human user. You have access to the following tool to help answer the question. <tool_description> Search Engine Tool * The search engine will exclusively search over Wikipedia for pages similar to your query. It returns for each page its title and full page content. Use this tool if you want to get up-to-date and comprehensive information on a topic to help answer queries. Queries should be as atomic as possible -- they only need to address one part of the user's question. For example, if the user's query is "what is the color of a basketball?", your search query should be "basketball". Here's another example: if the user's question is "Who created the first neural network?", your first query should be "neural network". As you can see, these queries are quite short. Think keywords, not phrases. * At any time, you can make a call to the search engine using the following syntax: <search_query>query_word</search_query>. * You'll then get results back in <search_result> tags.</tool_description>"""  # noqa: E501

retriever = WikipediaRetriever()

# This should be the same as the function name below
RETRIEVER_TOOL_NAME = "search"


@tool
def search(query):
    """Search with the retriever."""
    return retriever.get_relevant_documents(query)
