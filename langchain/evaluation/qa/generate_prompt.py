# flake8: noqa
import re
from typing import Dict

from langchain.prompts import PromptTemplate
from langchain.prompts.base import BaseOutputParser


class QAGenerationOutputParser(BaseOutputParser):
    """Parse output in question/answer pair."""

    def parse(self, text: str) -> Dict[str, str]:
        regex = r"QUESTION: (.*?)\nANSWER: (.*)"
        match = re.search(regex, text)
        if match:
            question = match.group(1)
            answer = match.group(2)
            return {"query": question, "answer": answer}
        else:
            raise ValueError(f"Could not parse output: {text}")


template = """You are a teacher coming up with questions to ask on a quiz. 
Given the following document, please generate a question and answer based on that document.

Example Format:
<Begin Document>
...
<End Document>
QUESTION: question here
ANSWER: answer here

These questions should be detailed and be based explicitly on information in the document. Begin!

<Begin Document>
{doc}
<End Document>"""
PROMPT = PromptTemplate(
    input_variables=["doc"], template=template, output_parser=QAGenerationOutputParser()
)
