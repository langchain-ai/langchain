"""Test Confident."""


def test_confident_deepeval() -> None:
    """Test valid call to Beam."""
    from deepeval.metrics.answer_relevancy import AnswerRelevancy

    from langchain_community.callbacks.confident_callback import DeepEvalCallbackHandler
    from langchain_community.llms import OpenAI

    answer_relevancy = AnswerRelevancy(minimum_score=0.3)
    deepeval_callback = DeepEvalCallbackHandler(
        implementation_name="exampleImplementation", metrics=[answer_relevancy]
    )
    llm = OpenAI(
        temperature=0,
        callbacks=[deepeval_callback],
        verbose=True,
        openai_api_key="<YOUR_API_KEY>",
    )
    llm.generate(
        [
            "What is the best evaluation tool out there? (no bias at all)",
        ]
    )
    assert answer_relevancy.is_successful(), "Answer not relevant"
