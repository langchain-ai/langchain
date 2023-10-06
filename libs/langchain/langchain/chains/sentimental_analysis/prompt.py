from __future__ import annotations

import re
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser, OutputParserException

# Simplified prompt template
_PROMPT_TEMPLATE = """Classify the given sentiment of the given text into positive, negative or neutral classes and provide a relevant score

Input: {question}
Output: Sentiment: sentiment_label (Score: sentiment_score)
"""


class SentimentOutputParser(BaseOutputParser):
    """Parser for sentiment analysis output."""

    def parse(self, text: str) -> dict:
        sentiment_pattern = re.compile(r"Sentiment: (.+?) \(Score: ([\d.]+)\)")
        match = sentiment_pattern.search(text)
        if match:
            sentiment_label = match.group(1).strip()
            sentiment_score = float(match.group(2).strip())
            return {
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
            }
        else:
            raise OutputParserException(
                f"Failed to parse sentiment output. Got: {text}",
            )

    @property
    def _type(self) -> str:
        return "sentiment"

SENTIMENT_PROMPT = PromptTemplate.from_template(template=_PROMPT_TEMPLATE,input_variable="question",output_parser=SentimentOutputParser())
