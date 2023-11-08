import pytest

from langchain_experimental.chains.qa_with_references_and_verbatims import (
    retrieval,
)
from tests.integration_tests.chains.test_integration_qa_with_references import (
    ALL_CHAIN_TYPE,
    ALL_SAMPLES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    REDUCE_K_BELOW_MAX_TOKENS,
    _test_qa_with_reference_chain,
)


@pytest.mark.parametrize("chain_type", ALL_CHAIN_TYPE)
@pytest.mark.parametrize("provider,question", ALL_SAMPLES)
def test_qa_with_reference_chain(provider: str, question: str, chain_type: str) -> None:
    _test_qa_with_reference_chain(
        cls=retrieval.RetrievalQAWithReferencesAndVerbatimsChain,
        provider=provider,
        chain_type=chain_type,
        question=question,
        max_token=2000,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        kwargs={
            "reduce_k_below_max_tokens": REDUCE_K_BELOW_MAX_TOKENS,
        },
    )
