import os

from langchain_community.vectorstores import Vectara
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

if os.environ.get("VECTARA_CUSTOMER_ID", None) is None:
    raise Exception("Missing `VECTARA_CUSTOMER_ID` environment variable.")
if os.environ.get("VECTARA_CORPUS_ID", None) is None:
    raise Exception("Missing `VECTARA_CORPUS_ID` environment variable.")
if os.environ.get("VECTARA_API_KEY", None) is None:
    raise Exception("Missing `VECTARA_API_KEY` environment variable.")

# Setup the Vectara retriever with your Corpus ID and API Key

# note you can customize the retriever behavior by passing additional arguments:
# - k: number of results to return (defaults to 5)
# - lambda_val: the
#   [lexical matching](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching)
#   factor for hybrid search (defaults to 0.025)
# - filter: a [filter](https://docs.vectara.com/docs/common-use-cases/filtering-by-metadata/filter-overview)
#   to apply to the results (default None)
# - n_sentence_context: number of sentences to include before/after the actual matching
#   segment when returning results. This defaults to 2.
# - mmr_config: can be used to specify MMR mode in the query.
#   - is_enabled: True or False
#   - mmr_k: number of results to use for MMR reranking
#   - diversity_bias: 0 = no diversity, 1 = full diversity. This is the lambda
#     parameter in the MMR formula and is in the range 0...1
retriever = Vectara().as_retriever()

# RAG pipeline: we extract the summary from the RAG output, which is the last document
# (if summary is enabled)
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
