import re
from typing import Dict, List, Set, Tuple

import pytest
from langchain.schema import Document, OutputParserException

from langchain_experimental.chains.qa_with_references_and_verbatims.base import (
    QAWithReferencesAndVerbatimsChain,
)
from tests.unit_tests.chains.test_qa_with_references import (
    ALL_CHAIN_TYPE,
    CALLBACKS,
    compare_responses,
    init_llm,
    logger,
)


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
                        '{"response": "He eats apples, pears and carrots.", '
                        '"documents": [{"ids": ["_idx_1", "_idx_2"], '
                        '"verbatims": ['
                        '"He eats apples and plays football.", '
                        '"He eats pears.", '
                        '"He eats carrots."]}]}'
                        "```"
                    },
                    [
                        ["He eats\napples and plays football.", "He eats pears."],
                        [
                            "He eats carrots.",
                        ],
                    ],
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "map_reduce": (
                    {
                        0: 'Output: {"ids": [], "verbatims": []}',
                        1: "Output:\n"
                        '{"ids": ["_idx_1", "_idx_2"], '
                        '"verbatims": ["He eats apples", "He eats pears"]}',
                        2: "Output: \n"
                        '{"ids": ["_idx_0"], '
                        '"verbatims": ["He eats carrots."]}',
                        3: 'Output: {"ids": [], "verbatims": []}',
                        4: '{"response": "He eats apples, He eats pears, '
                        'He eats carrots.", "documents": '
                        '[{"ids": ["_idx_0"], "verbatims": '
                        '["He eats carrots."]}, '
                        '{"ids": ["_idx_1", "_idx_2"], '
                        '"verbatims": ["He eats apples", "He eats pears"]}]}',
                        5: "He eats apples, He eats pears, He eats carrots.",
                    },
                    [["eats apples", "eats pears"], ["eats carrots"]],
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "refine": (
                    {
                        0: "The output would be:\n"
                        '{"response": '
                        '"I don\'t know", '
                        '"documents": ['
                        '{"ids": ["_idx_0"], "verbatims": []}]}',
                        1: '{"response": "He eats apples and pears.", '
                        '"documents": [{"ids": ['
                        '"_idx_0", "_idx_1"], '
                        '"verbatims": ['
                        '"He eats apples and plays football.", '
                        '"He eats pears."]}]}',
                        2: "The output would be:\n"
                        '{"response": "He eats apples, pears, and carrots.", '
                        '"documents": ['
                        '{"ids": ["_idx_0", "_idx_1", "_idx_2", "_idx_3"], '
                        '"verbatims": ["He eats apples and plays football.", '
                        '"He eats pears.", "He eats carrots. '
                        'I like football.","The Earth is round."]}]}',
                        3: "The output would be:\n"
                        '{"response": "He eats apples, pears, and carrots.", '
                        '"documents": ['
                        '{"ids": ["_idx_0", "_idx_1", "_idx_2", "_idx_3"], '
                        '"verbatims": ['
                        '"He eats apples and plays football.", '
                        '"He eats pears.", '
                        '"He eats carrots. I like football.",'
                        '"The Earth is round."]}]}',
                    },
                    [
                        [],
                        # ["eats apples", "eats pears"],
                        ["eats carrots"],
                    ],
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "map_rerank": (
                    {
                        0: '{"response": "This document does not answer '
                        'the question", "documents": []}\n'
                        "Score: 0",
                        1: '{"response": "apples and pears", '
                        '"documents": [{"ids": ["99"], '
                        '"verbatims": ["He eats apples", '
                        '"He eats pears"]}]}\n'
                        "Score: 100",
                        2: '{"response": "carrots", '
                        '"documents": ['
                        '{"ids": ["99"], '
                        '"verbatims": ["He eats carrots"]}]}\n'
                        "Score: 100",
                        3: '{"response": "This document does not '
                        'answer the question", '
                        '"documents": []}\n'
                        "Score: 0",
                        4: ' {"response": "apples and pears", '
                        '"documents": [{"ids": ["99"], '
                        '"verbatims": ["He eats apples", '
                        '"He eats pears"]}]}',
                    },
                    [
                        ["He eats\napples", "He eats pears."],
                    ],
                    r"(?i).*\bapples\b.*\bpears",
                    {1},
                ),
            },
        ),
    ],
)
@pytest.mark.parametrize("chain_type", ALL_CHAIN_TYPE)
def test_qa_with_reference_and_verbatims_chain(
    question: str,
    docs: List[Document],
    map_responses: Dict[str, Tuple[Dict[int, str], List[List[str]], str, Set[int]]],
    chain_type: str,
) -> None:
    # chain_type = "map_reduce"  # stuff, map_reduce, refine, map_rerank

    queries, verbatims, expected_answer, references = map_responses[chain_type]
    llm = init_llm(queries)

    for i in range(0, 2):  # Retry if empty ?
        try:
            qa_chain = QAWithReferencesAndVerbatimsChain.from_chain_type(
                llm=llm,
                chain_type=chain_type,
                original_verbatim=True,
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
                llm.cache = False
                continue
            assert re.match(expected_answer, answer_of_question)
            for ref, original, assert_verbatims in zip(
                references, answer["source_documents"], verbatims
            ):
                assert docs[ref] is original, "Return incorrect original document"
                assert compare_responses(
                    original.metadata.get("verbatims", []), assert_verbatims
                ), "Return incorrect verbatims"
            break
        except OutputParserException as e:
            llm.cache = False
            logger.warning("Parsing error. Retry", e)
            continue  # Retry
    else:
        print(f"Response is Empty after {i + 1} tries.")
        assert False, "Impossible to receive a response"
    print(f"response après {i}")
