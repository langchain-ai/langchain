import pytest
from langchain.chains import RetrievalQAWithSourcesChain

from tests.integration_tests.chains.test_integration_qa_with_references import (
    ALL_CHAIN_TYPE,
    ALL_SAMPLES,
    REDUCE_K_BELOW_MAX_TOKENS,
    _test_qa_with_reference_chain,
)


@pytest.mark.parametrize("chain_type", ALL_CHAIN_TYPE)
# @pytest.mark.parametrize("provider,question",
#                          sorted({(k, l) for k, ls in samples.items() for l in ls}))
@pytest.mark.parametrize("provider,question", ALL_SAMPLES)
def test_qa_with_sources_chain(provider: str, question: str, chain_type: str) -> None:
    _test_qa_with_reference_chain(
        cls=RetrievalQAWithSourcesChain,
        provider=provider,
        chain_type=chain_type,
        question=question,
        max_token=1000,
        chunk_size=200,
        chunk_overlap=0,
        kwargs={
            "return_source_documents": True,
            "reduce_k_below_max_tokens": REDUCE_K_BELOW_MAX_TOKENS,
        },
    )
