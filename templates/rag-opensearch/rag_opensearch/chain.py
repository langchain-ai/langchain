import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL= os.getenv("OPENSEARCH_URL")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD= os.getenv("OPENSEARCH_PASSWORD")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME")


vector_store = OpenSearchVectorSearch(
    opensearch_url=OPENSEARCH_URL,
    index_name=OPENSEARCH_INDEX_NAME,
    
    
)

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who speaks like a pirate",
        ),
        ("human", "{text}"),
    ]
)
_model = ChatOpenAI()

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model
