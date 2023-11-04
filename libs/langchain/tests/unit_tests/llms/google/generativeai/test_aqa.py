from unittest.mock import MagicMock, patch

import pytest

try:
    import google.ai.generativelanguage as genai

    has_google = True
except ImportError:
    has_google = False

from langchain.llms.google.generativeai import (
    AqaModelInput,
    GenAIAqa,
)

if has_google:
    import langchain.vectorstores.google.generativeai.genai_extension as genaix

    # Make sure the tests do not hit actual production servers.
    genaix.set_defaults(
        genaix.Config(api_endpoint="No-such-endpoint-to-prevent-hitting-real-backend")
    )


@pytest.mark.requires("google.ai.generativelanguage")
def test_it_can_be_constructed() -> None:
    GenAIAqa()


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.TextServiceClient.generate_text_answer")
def test_invoke(mock_generate_text_answer: MagicMock) -> None:
    # Arrange
    mock_generate_text_answer.return_value = genai.GenerateTextAnswerResponse(
        answer=genai.TextCompletion(
            output="42",
            citation_metadata=genai.CitationMetadata(
                citation_sources=[
                    genai.CitationSource(
                        start_index=100,
                        end_index=200,
                        uri="answer.com/meaning_of_life.txt",
                    )
                ]
            ),
        ),
        attributed_passages=[
            genai.AttributedPassage(
                text="Meaning of life is 42.",
                passage_ids=["corpora/123/documents/456/chunks/789"],
            ),
        ],
        answerable_probability=0.7,
    )

    # Act
    aqa = GenAIAqa(answer_style=genai.AnswerStyle.EXTRACTIVE)
    output = aqa.invoke(
        input=AqaModelInput(
            prompt="What is the meaning of life?",
            source_passages=["It's 42."],
        )
    )

    # Assert
    assert output.answer == "42"
    assert output.attributed_passages == ["Meaning of life is 42."]
    assert output.answerable_probability == pytest.approx(0.7)

    assert mock_generate_text_answer.call_count == 1
    request = mock_generate_text_answer.call_args.args[0]
    assert request.question.text == "What is the meaning of life?"
    assert request.answer_style == genai.AnswerStyle.EXTRACTIVE
    passages = request.grounding_source.passages.passages
    assert len(passages) == 1
    passage = passages[0]
    assert passage.text == "It's 42."
