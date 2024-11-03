import os

from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores.vectara import SummaryConfig, VectaraQueryConfig
from langchain_core.pydantic_v1 import BaseModel

if os.environ.get("VECTARA_CUSTOMER_ID", None) is None:
    raise Exception("Missing `VECTARA_CUSTOMER_ID` environment variable.")
if os.environ.get("VECTARA_CORPUS_ID", None) is None:
    raise Exception("Missing `VECTARA_CORPUS_ID` environment variable.")
if os.environ.get("VECTARA_API_KEY", None) is None:
    raise Exception("Missing `VECTARA_API_KEY` environment variable.")

# Setup the Vectara vectorstore with your Corpus ID and API Key
vectara = Vectara()

# Define the query configuration:
summary_config = SummaryConfig(is_enabled=True, max_results=5, response_lang="eng")
config = VectaraQueryConfig(k=10, lambda_val=0.005, summary_config=summary_config)

rag = Vectara().as_rag(config)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = rag.with_types(input_type=Question)
