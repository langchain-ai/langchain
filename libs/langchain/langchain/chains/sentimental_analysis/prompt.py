import re
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.output import LLMResult, Generation
from langchain.schema import OutputParserException
from langchain.schema import BaseOutputParser

_PROMPT_TEMPLATE = """
Classify the given sentiment of the given text into positive, negative or neutral classes and provide a relevant score

Input: {question}
Output: Sentiment: sentiment_label (Score: sentiment_score)
"""


class SentimentOutputParser(BaseOutputParser):
    """Parser for sentiment analysis output."""

    def parse(self, text: str) -> LLMResult:
        sentiment_results = []

        sentiment_pattern = re.compile(r"Sentiment: (.+?) \(Score: ([\d.]+)\)")
        matches = sentiment_pattern.findall(text)

        for match in matches:
            sentiment_label = match[0].strip()
            sentiment_score = float(match[1].strip())

            generation = Generation(
                text=f"Sentiment Label: {sentiment_label}\nSentiment Score: {sentiment_score}",
                metadata={
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                },
            )

            sentiment_results.append(generation)

        if not sentiment_results:
            raise OutputParserException(
                f"Failed to parse sentiment output. Got: {text}"
            )

        return LLMResult(candidates=sentiment_results)

    @property
    def _type(self) -> str:
        return "sentiment"


SENTIMENT_PROMPT = PromptTemplate.from_template(
    template=_PROMPT_TEMPLATE,
    input_variable=["question"],
    output_parser=SentimentOutputParser(),
)
