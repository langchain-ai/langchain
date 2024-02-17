import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"],
    deployment=os.environ["AZURE_EMBEDDINGS_DEPLOYMENT"],
    chunk_size=1
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    azure_search_key=os.environ["AZURE_SEARCH_KEY"],
    index_name="rag-azure-search",
    embedding_function=embeddings.embed_query,
    search_type="similarity"
)

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# loader = TextLoader("/home/krpratic/langchain/templates/rag-azure-search/data/state_of_the_union.txt", encoding="utf-8")

# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# vector_store.add_documents(documents=docs)

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""

# Perform a similarity search
retriever = vector_store.as_retriever()

_prompt = ChatPromptTemplate.from_template(template)
_model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["AZURE_CHAT_DEPLOYMENT"],
)
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | _prompt
    | _model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
