import os
from typing import List, Optional

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores.vectara import Vectara
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from rag_vectara_selfquery import defaults

if os.environ.get("VECTARA_CUSTOMER_ID", None) is None:
    raise Exception("Missing `VECTARA_CUSTOMER_ID` environment variable.")
if os.environ.get("VECTARA_CORPUS_ID", None) is None:
    raise Exception("Missing `VECTARA_CORPUS_ID` environment variable.")
if os.environ.get("VECTARA_API_KEY", None) is None:
    raise Exception("Missing `VECTARA_API_KEY` environment variable.")

def create_chain(
    llm: Optional[BaseLLM] = None,
    document_contents: str = defaults.DEFAULT_DOCUMENT_CONTENTS,
    metadata_field_info: List[AttributeInfo] = defaults.DEFAULT_METADATA_FIELD_INFO,
):
    """
    Create a chain that can be used to query a Qdrant vector store with a self-querying
    capability. By default, this chain will use the OpenAI LLM and OpenAIEmbeddings, and
    work with the default document contents and metadata field info. You can override
    these defaults by passing in your own values.
    :param llm: an LLM to use for generating text
    :param embeddings: an Embeddings to use for generating queries
    :param document_contents: a description of the document set
    :param metadata_field_info: list of metadata attributes
    :param collection_name: name of the Qdrant collection to use
    :return:
    """
    llm = llm or OpenAI(temperature=0)

    # Setup the Vectara instance
    # note you can customize the retriever behavior by passing additional arguments:
    # - k: number of results to return (defaults to 5)
    # - lambda_val: the
    #   [lexical matching](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching)
    #   factor for hybrid search (defaults to 0.025)
    # - filter: a [filter](https://docs.vectara.com/docs/common-use-cases/filtering-by-metadata/filter-overview)
    #   to apply to the results (default None)
    # - n_sentence_context: number of sentences to include before/after the actual
    #   matching segment when returning results. This defaults to 2.
    # - mmr_config: can be used to specify MMR mode in the query.
    #   - is_enabled: True or False
    #   - mmr_k: number of results to use for MMR reranking
    #   - diversity_bias: 0 = no diversity, 1 = full diversity. This is the lambda
    #     parameter in the MMR formula and is in the range 0...1
    vectara = Vectara()

    # Set up a retriever to query your vectara service with self-querying capabilities
    retriever = SelfQueryRetriever.from_llm(
        llm, vectara, document_contents, metadata_field_info, verbose=True
    )

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | (lambda res: res[-1])
        | StrOutputParser()
    )
    return chain

chain = create_chain()
