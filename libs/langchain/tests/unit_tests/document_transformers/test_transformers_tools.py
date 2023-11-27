from typing import Dict

import pytest

from langchain.document_transformers.copy_transformer import CopyDocumentTransformer
from langchain.document_transformers.generate_questions import (
    GenerateQuestionsTransformer,
)
from langchain.document_transformers.sumarize_and_questions_transformer import (
    SummarizeAndQuestionsTransformer,
)
from langchain.document_transformers.sumarize_transformer import (
    SummarizeTransformer,
)

# Note: Import directly from langchain_core is not stable and generate some errors
# from langchain_core.language_models import BaseLLM
# from langchain_core.documents import Document
from langchain.llms import BaseLLM
from langchain.schema import Document
from tests.unit_tests.llms.fake_llm import FakeLLM

TEMPERATURE = 0.0
MAX_TOKENS = 1000
FAKE_LLM = True
USE_CACHE = True


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
        from dotenv import load_dotenv

        import langchain
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


# %% copy_transformer
def test_copy_transformer_transform_documents() -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    result = CopyDocumentTransformer().transform_documents([doc1, doc2])
    assert len(result) == 2
    assert id(result[0]) != id(doc1)
    assert id(result[1]) != id(doc2)
    assert result[0] == doc1
    assert result[1] == doc2


def test_copy_transformer_lazy_transform_documents() -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    result = list(
        CopyDocumentTransformer().lazy_transform_documents(iter([doc1, doc2]))
    )
    assert len(result) == 2
    assert id(result[0]) != id(doc1)
    assert id(result[1]) != id(doc2)
    assert result[0] == doc1
    assert result[1] == doc2


@pytest.mark.asyncio
async def test_copy_transformer_atransform_documents() -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    result = await CopyDocumentTransformer().atransform_documents([doc1, doc2])
    assert len(result) == 2
    assert id(result[0]) != id(doc1)
    assert id(result[1]) != id(doc2)
    assert result[0] == doc1
    assert result[1] == doc2


@pytest.mark.asyncio
async def test_copy_transformer_alazy_transform_documents() -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    result = [
        doc
        async for doc in CopyDocumentTransformer().alazy_transform_documents(
            iter([doc1, doc2])
        )
    ]
    assert len(result) == 2
    assert id(result[0]) != id(doc1)
    assert id(result[1]) != id(doc2)
    assert result[0] == doc1
    assert result[1] == doc2


# %% generate_questions
def test_generate_questions_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used "
            "in the past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 6


def test_generate_questions_lazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, 
    formulas and related structures, shapes and the spaces in which they are 
    contained, and quantities and their changes. These topics are represented 
    in modern mathematics with the major subdisciplines of number theory, algebra, 
    geometry, and analysis, respectively. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used in the "
            "past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = list(transformer.lazy_transform_documents(iter([doc1, doc2])))
    assert len(result) == 6


@pytest.mark.asyncio
async def test_generate_questions_atransform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation "
            "used in the past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = await transformer.atransform_documents([doc1, doc2])
    assert len(result) == 6


@pytest.mark.asyncio
async def test_generate_questions_alazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used in the "
            "past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 6


# %% sumarize_transformer


def test_sumarize_transformer_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "SUMMARY:\nMathematics is the study of numbers, formulas, shapes, "
            "spaces, "
            "quantities, and their changes.",
            1: "SUMMARY:\nMathematics has a long history of discoveries and the "
            "development "
            "of mathematical methods and notation.",
        }
    )
    transformer = SummarizeTransformer.from_llm(llm=llm)
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 2


def test_sumarize_transformer_lazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "SUMMARY:\nMathematics is the study of numbers, formulas, "
            "shapes, spaces, "
            "quantities, and their changes.",
            1: "SUMMARY:\nMathematics has a long history of discoveries and "
            "the development "
            "of mathematical methods and notation.",
        }
    )
    transformer = SummarizeTransformer.from_llm(llm=llm)
    result = list(transformer.lazy_transform_documents(iter([doc1, doc2])))
    assert len(result) == 2


@pytest.mark.asyncio
async def test_sumarize_transformer_atransform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "SUMMARY:\nMathematics is the study of numbers, formulas, shapes, "
            "spaces, "
            "quantities, and their changes.",
            1: "SUMMARY:\nMathematics has a long history of discoveries and "
            "the development "
            "of mathematical methods and notation.",
        }
    )
    transformer = SummarizeTransformer.from_llm(llm=llm)
    result = await transformer.atransform_documents([doc1, doc2])
    assert len(result) == 2


@pytest.mark.asyncio
async def test_sumarize_transformer_alazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "SUMMARY:\nMathematics is the study of numbers, formulas, "
            "shapes, spaces, "
            "quantities, and their changes.",
            1: "SUMMARY:\nMathematics has a long history of discoveries "
            "and the development "
            "of mathematical methods and notation.",
        }
    )
    transformer = SummarizeTransformer.from_llm(llm=llm)
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 2


# %% sumarize_and_questions_transformer


def test_sumarize_and_questions_transformer_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "Output:\n"
            "```\n"
            "{\n"
            '    "summary": "Mathematics is an area of knowledge that includes topics '
            "such as numbers, formulas, shapes, spaces, quantities, and their "
            'changes.",\n'
            '    "questions": [\n'
            '        "What topics are included in mathematics?",\n'
            '        "What are some examples of mathematical structures?",\n'
            '        "How do quantities and their changes relate to mathematics?"\n'
            "    ]\n"
            "}\n"
            "```\n",
            1: "Output:\n"
            "    ```\n"
            "    {\n"
            '        "summary": "The history of mathematics deals with the origin of '
            "discoveries in mathematics and the mathematical methods and notation "
            'of the past.",\n'
            '        "questions": [\n'
            '            "What is the history of mathematics?",\n'
            '            "What discoveries have been made in mathematics?",\n'
            '           "What are the mathematical methods and notation of the past?"\n'
            "        ]\n"
            "    }\n"
            "    ```\n",
        }
    )
    transformer = SummarizeAndQuestionsTransformer.from_llm(llm=llm)
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 8


def test_sumarize_and_questions_transformer_lazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "Output:\n"
            "```\n"
            "{\n"
            '    "summary": "Mathematics is an area of knowledge that includes topics '
            "such as numbers, formulas, shapes, spaces, quantities, and their "
            'changes.",\n'
            '    "questions": [\n'
            '        "What topics are included in mathematics?",\n'
            '        "What are some examples of mathematical structures?",\n'
            '        "How do quantities and their changes relate to mathematics?"\n'
            "    ]\n"
            "}\n"
            "```\n",
            1: "Output:\n"
            "    ```\n"
            "    {\n"
            '        "summary": "The history of mathematics deals with the origin '
            "of discoveries in mathematics and the mathematical methods and "
            'notation of the past.",\n'
            '        "questions": [\n'
            '            "What is the history of mathematics?",\n'
            '            "What discoveries have been made in mathematics?",\n'
            '           "What are the mathematical methods and notation of the past?"\n'
            "        ]\n"
            "    }\n"
            "    ```\n",
        }
    )
    transformer = SummarizeAndQuestionsTransformer.from_llm(llm=llm)
    result = list(transformer.lazy_transform_documents(iter([doc1, doc2])))
    assert len(result) == 8


@pytest.mark.asyncio
async def test_sumarize_and_questions_transformer_atransform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "Output:\n"
            "```\n"
            "{\n"
            '    "summary": "Mathematics is an area of knowledge that includes topics '
            "such as numbers, formulas, shapes, spaces, quantities, and their "
            'changes.",\n'
            '    "questions": [\n'
            '        "What topics are included in mathematics?",\n'
            '        "What are some examples of mathematical structures?",\n'
            '        "How do quantities and their changes relate to mathematics?"\n'
            "    ]\n"
            "}\n"
            "```\n",
            1: "Output:\n"
            "    ```\n"
            "    {\n"
            '        "summary": "The history of mathematics deals with the origin of '
            "discoveries in mathematics and the mathematical methods and notation "
            'of the past.",\n'
            '        "questions": [\n'
            '            "What is the history of mathematics?",\n'
            '            "What discoveries have been made in mathematics?",\n'
            '           "What are the mathematical methods and notation of the past?"\n'
            "        ]\n"
            "    }\n"
            "    ```\n",
        }
    )
    transformer = SummarizeAndQuestionsTransformer.from_llm(llm=llm)
    result = await transformer.atransform_documents([doc1, doc2])
    assert len(result) == 8


@pytest.mark.asyncio
async def test_sumarize_and_questions_transformer_alazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "Output:\n"
            "```\n"
            "{\n"
            '    "summary": "Mathematics is an area of knowledge that includes topics '
            "such as numbers, formulas, shapes, spaces, quantities, "
            'and their changes.",\n'
            '    "questions": [\n'
            '        "What topics are included in mathematics?",\n'
            '        "What are some examples of mathematical structures?",\n'
            '        "How do quantities and their changes relate to mathematics?"\n'
            "    ]\n"
            "}\n"
            "```\n",
            1: "Output:\n"
            "    ```\n"
            "    {\n"
            '        "summary": "The history of mathematics deals with the origin '
            "of discoveries in mathematics and the mathematical methods and "
            'notation of the past.",\n'
            '        "questions": [\n'
            '            "What is the history of mathematics?",\n'
            '            "What discoveries have been made in mathematics?",\n'
            '           "What are the mathematical methods and notation of the past?"\n'
            "        ]\n"
            "    }\n"
            "    ```\n",
        }
    )
    transformer = SummarizeAndQuestionsTransformer.from_llm(llm=llm)
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 8
