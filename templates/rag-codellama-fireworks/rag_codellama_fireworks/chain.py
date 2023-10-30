import os

from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms.fireworks import Fireworks
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Check API key
if os.environ.get("FIREWORKS_API_KEY", None) is None:
    raise Exception("Missing `FIREWORKS_API_KEY` environment variable.")

# Load codebase
# Set local path
repo_path = "/Users/rlm/Desktop/tmp_repo"
# Use LangChain as an example
repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)
loader = GenericLoader.from_filesystem(
    repo_path + "/libs/langchain/langchain",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()

# Split
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=texts,
    collection_name="codebase-rag",
    embedding=GPT4AllEmbeddings(),
)
retriever = vectorstore.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize a Fireworks model
model = Fireworks(model="accounts/fireworks/models/llama-v2-34b-code-instruct")

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
