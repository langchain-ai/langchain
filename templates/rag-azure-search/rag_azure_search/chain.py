import os

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    raise ValueError("Please set the environment variable AZURE_OPENAI_ENDPOINT")

if not os.getenv("AZURE_OPENAI_API_KEY"):
    raise ValueError("Please set the environment variable AZURE_OPENAI_API_KEY")

if not os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"):
    raise ValueError("Please set the environment variable AZURE_EMBEDDINGS_DEPLOYMENT")

if not os.getenv("AZURE_CHAT_DEPLOYMENT"):
    raise ValueError("Please set the environment variable AZURE_CHAT_DEPLOYMENT")

if not os.getenv("AZURE_SEARCH_ENDPOINT"):
    raise ValueError("Please set the environment variable AZURE_SEARCH_ENDPOINT")

if not os.getenv("AZURE_SEARCH_KEY"):
    raise ValueError("Please set the environment variable AZURE_SEARCH_KEY")


api_version = os.getenv("OPENAI_API_VERSION", "2023-05-15")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-azure-search")

embeddings = AzureOpenAIEmbeddings(
    deployment=os.environ["AZURE_EMBEDDINGS_DEPLOYMENT"],
    api_version=api_version,
    chunk_size=1,
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    azure_search_key=os.environ["AZURE_SEARCH_KEY"],
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

"""
(Optional) Example document - 
Uncomment the following code to load the document into the vector store
or substitute with your own.
"""
# import pathlib
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# current_file_path = pathlib.Path(__file__).resolve()
# root_directory = current_file_path.parents[3]
# target_file_path = \
#     root_directory / "docs" / "docs" / "modules" / "state_of_the_union.txt"

# loader = TextLoader(str(target_file_path), encoding="utf-8")

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
    deployment_name=os.environ["AZURE_CHAT_DEPLOYMENT"],
    api_version=api_version,
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
