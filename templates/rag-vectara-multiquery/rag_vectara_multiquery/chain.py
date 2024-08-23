import os

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores.vectara import SummaryConfig, VectaraQueryConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI

if os.environ.get("VECTARA_CUSTOMER_ID", None) is None:
    raise Exception("Missing `VECTARA_CUSTOMER_ID` environment variable.")
if os.environ.get("VECTARA_CORPUS_ID", None) is None:
    raise Exception("Missing `VECTARA_CORPUS_ID` environment variable.")
if os.environ.get("VECTARA_API_KEY", None) is None:
    raise Exception("Missing `VECTARA_API_KEY` environment variable.")


# Setup the Vectara retriever with your Corpus ID and API Key
vectara = Vectara()

# Define the query configuration:
summary_config = SummaryConfig(is_enabled=True, max_results=5, response_lang="eng")
config = VectaraQueryConfig(k=10, lambda_val=0.005, summary_config=summary_config)

# Setup the Multi-query retriever
llm = ChatOpenAI(temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectara.as_retriever(config=config), llm=llm
)

# Setup RAG pipeline with multi-query.
# We extract the summary from the RAG output, which is the last document in the list.
# Note that if you want to extract the citation information, you can use res[:-1]]
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | (lambda res: res[-1])
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
