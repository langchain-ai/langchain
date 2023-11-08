import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import Callbacks
from langchain.llms import BaseLLM
from langchain.schema import Document, OutputParserException

from langchain_experimental.chains.qa_with_references.base import (
    QAWithReferencesChain,
)
from tests.unit_tests.fake_llm import FakeLLM

FAKE_LLM = True
VERBOSE_PROMPT = False
VERBOSE_RESULT = False
USE_CACHE = True
CHUNK_SIZE = 500
CHUNK_OVERLAP = 5
TEMPERATURE = 0.0
MAX_TOKENS = 2000
ALL_CHAIN_TYPE = ["stuff", "map_reduce", "refine", "map_rerank"]

CALLBACKS: Callbacks = []

logger = logging.getLogger(__name__)

if VERBOSE_PROMPT or VERBOSE_RESULT:

    class ExStdOutCallbackHandler(StdOutCallbackHandler):
        def on_text(
            self,
            text: str,
            color: Optional[str] = None,
            end: str = "",
            **kwargs: Any,
        ) -> None:
            if VERBOSE_PROMPT:
                print("====")
                super().on_text(text=text, color=color, end=end)

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
            """Ajoute une trace des outputs du llm"""
            if VERBOSE_RESULT:
                print("\n\033[1m> Finished chain with\033[0m")
                knows_keys = {
                    "answer",
                    "output_text",
                    "text",
                    "result",
                    "outputs",
                    "output",
                }
                if "outputs" in outputs:
                    print("\n\033[33m")
                    print(
                        "\n---\n".join(
                            [text["text"].strip() for text in outputs["outputs"]]
                        )
                    )
                    print("\n\033[0m")
                elif knows_keys.intersection(outputs):
                    # Prend la première cles en intersection
                    print(
                        f"\n\033[33m{outputs[next(iter(knows_keys.intersection(outputs)))]}\n\033[0m"
                    )
                else:
                    pass

    CALLBACKS = [ExStdOutCallbackHandler()]


def init_llm(
    queries: Dict[int, str],
    max_token: int = MAX_TOKENS,
) -> BaseLLM:
    if FAKE_LLM:
        return FakeLLM(
            queries=queries,
            sequential_responses=True,
        )
    else:
        import langchain
        from dotenv import load_dotenv
        from langchain.cache import SQLiteCache

        load_dotenv()

        if USE_CACHE:
            langchain.llm_cache = SQLiteCache(
                database_path="/tmp/cache_qa_with_reference.db"
            )
        llm = langchain.OpenAI(
            temperature=TEMPERATURE,
            max_tokens=max_token,
            # cache=False,
        )
        return llm


def compare_words_of_responses(response: str, assert_response: str) -> bool:
    """The exact format of verbatim may be changed by the LLM.
    Extract only the words of the verbatim, and try to find a sequence
    of same words in the original document.
    """
    only_words = filter(len, re.split(r"[^\w]+", assert_response))
    regex_for_words_in_same_oder = (
        r"(?i)\b" + r"\b[^\w]+".join(only_words) + r"\b" r"\s*[.!?:;]?"
    )
    match = re.search(regex_for_words_in_same_oder, response, re.IGNORECASE)
    if match:
        return True
    return False  # No verbatim found in the original document


def compare_responses(responses: List[str], assert_responses: List[str]) -> bool:
    for response, assert_response in zip(responses, assert_responses):
        if not compare_words_of_responses(response, assert_response):
            return False
    return True


@pytest.mark.parametrize(
    "question,docs,map_responses",
    [
        (
            "what does he eat?",
            [
                Document(
                    page_content="The night is black.",
                    metadata={},
                ),
                Document(
                    page_content="He eats\napples and plays football. "
                    "My name is Philippe. He eats pears.",
                    metadata={},
                ),
                Document(
                    page_content="He eats carrots. I like football.",
                    metadata={},
                ),
                Document(
                    page_content="The Earth is round.",
                    metadata={},
                ),
            ],
            {
                "stuff": (
                    {
                        0: "```\n"
                        "He eats apples, pears and carrots.\n"
                        "IDS: _idx_1, _idx_2\n"
                        "```\n",
                    },
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "map_reduce": (
                    {
                        0: 'Output: {"lines": []}',
                        1: 'Output: {"lines": ["_idx_1: He eats apples", '
                        '"_idx_3: He eats pears"]}',
                        2: 'Output: {"lines": ["_idx_1: He eats carrots."]}',
                        3: 'Output: {"lines": []}',
                        4: " He eats apples, pears, and carrots.\n"
                        "IDS: _idx_1, _idx_3, _idx_2\n",
                    },
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "refine": (
                    {
                        0: "Answer: I don't know.\nIDS:\n",
                        1: "Answer: He eats apples, pears "
                        "and plays football.\nIDS: _idx_1\n",
                        2: "Answer: He eats apples, pears, carrots "
                        "and plays football.\nIDS: _idx_1, _idx_2\n",
                        3: "Answer: He eats apples, pears, carrots "
                        "and plays football.\nIDS: _idx_1, _idx_2, _idx_3\n",
                    },
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "map_rerank": (
                    {
                        0: "This document does not answer the question\n" "Score: 0\n",
                        1: "apples and pears\nScore: 100\n",
                        2: "carrots\nScore: 100\n",
                        3: "This document does not answer the question.\n" "Score: 0\n",
                        4: "apples and pears\n",
                    },
                    r"(?i).*\bapples\b.*\bpears",
                    {1},
                ),
            },
        ),
    ],
)
@pytest.mark.parametrize("chain_type", ALL_CHAIN_TYPE)
def test_qa_with_reference_chain(
    question: str,
    docs: List[Document],
    map_responses: Dict[str, Tuple[Dict[int, str], str, Set[int]]],
    chain_type: str,
) -> None:
    queries, expected_answer, references = map_responses[chain_type]
    llm = init_llm(queries)

    for i in range(0, 2):  # Retry if error ?
        try:
            qa_chain = QAWithReferencesChain.from_chain_type(
                llm=llm,
                chain_type=chain_type,
            )
            answer = qa_chain(
                inputs={
                    "docs": docs,
                    "question": question,
                },
                callbacks=CALLBACKS,
            )
            answer_of_question = answer["answer"]
            if not answer_of_question:
                logger.warning("Return nothing. Retry")
                continue
            assert re.match(expected_answer, answer_of_question)
            for ref, original in zip(references, answer["source_documents"]):
                assert docs[ref] is original, "Return incorrect original document"
            break
        except OutputParserException:
            llm.cache = False
            logger.warning("Parsing error. Retry")
            continue  # Retry

    else:
        print(f"response after {i + 1} tries.")
        assert not "Impossible to receive a correct response"
