import re
from typing import Dict, List, Set, Tuple

import pytest
from langchain.chains import QAWithSourcesChain
from langchain.schema import Document, OutputParserException

from tests.unit_tests.chains.test_qa_with_references import (
    ALL_CHAIN_TYPE,
    CALLBACKS,
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
                    page_content="He eats\napples and plays football. "
                    "My name is Philippe. He eats pears.",
                    metadata={"source": "http:www.sample.fr/2"},
                ),
                Document(
                    page_content="The night is black.",
                    metadata={"source": "http:www.sample.fr/1"},
                ),
                Document(
                    page_content="He eats carrots. I like football.",
                    metadata={"source": "http:www.sample.fr/3"},
                ),
                Document(
                    page_content="The Earth is round.",
                    metadata={"source": "http:www.sample.fr/1"},
                ),
            ],
            {
                "stuff": (
                    {
                        0: " He eats apples, pears and carrots.\n"
                        "SOURCES: http:www.sample.fr/2, http:www.sample.fr/3"
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
                        0: "None",
                        1: "He eats apples and pears.",
                        2: "He eats carrots.",
                        3: "None",
                        4: " He eats apples, pears and carrots.\n"
                        "SOURCES: http:www.sample.fr/2, http:www.sample.fr/3",
                    },
                    [["eats apples", "eats pears"], ["eats carrots"]],
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "refine": (
                    {
                        0: "He eats apples and pears.",
                        1: "He eats apples, pears, and other fruits.",
                        2: '"He eats apples, pears, carrots, and other fruits.\n'
                        "Source: http:www.sample.fr/3"
                        'I like football.","The Earth is round."]}]}',
                        3: "He eats apples, pears, carrots, and other fruits "
                        "and vegetables.\n"
                        "Source: http:www.sample.fr/3",
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
def test_qa_with_sources_chain(
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
            qa_chain = QAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type=chain_type,
                return_source_documents=True,
            )
            answer = qa_chain(
                inputs={
                    "docs": docs,
                    "question": question,
                },
                callbacks=CALLBACKS,
            )
            answer_of_question = answer["answer"].strip()
            if not answer_of_question:
                logger.warning("Return nothing. Retry")
                llm.cache = False
                continue

            # Old QA with sources
            print(f'Source "{answer["sources"]}"')
            for doc in answer.get("source_documents", []):
                print(f'- Doc {doc.metadata["source"]}')

            assert re.match(expected_answer, answer_of_question)

            break
        except OutputParserException:
            llm.cache = False
            logger.warning("Parsing error. Retry")
            continue  # Retry
    else:
        print(f"Response is Empty after {i + 1} tries.")
        assert False, "Impossible to receive a response"
    print(f"response après {i}")
