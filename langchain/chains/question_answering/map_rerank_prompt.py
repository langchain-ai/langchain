from langchain.prompts import PromptTemplate
from langchain.prompts.base import BaseOutputParser
import re

template = """Use the following document to answer the given question. In addition to providing an answer, please also give your answer a score from 0-100 in terms of how good it is (higher is better). 

What decides the score? A good score is factually accurate, and FULLY answers the question in a way the user would find helpful. If the document does not contain the answer, the score should be 0. You should only give a score of 100 if you are absolutely positive this is the best answer. Keep in mind that you will also be answering this question with other documents, so one of them could have a better answer.

Use the following format:

Document:
---------------
Document text here
---------------
Question: Question here
Answer: Answer here
Score: Score (between 0 and 100) here

Begin!

Document:
---------------
{context}
---------------
Question: {question}
Answer:"""


class ScoreOutputParser(BaseOutputParser):

    def parse(self, text: str):
        regex = r"(.*?)\nScore: (.*)"
        match = re.search(regex, text)
        if match:
            question = match.group(1)
            answer = match.group(2)
            return {"answer": question, "score": int(answer)}
        else:
            raise ValueError(f"Could not parse output: {text}")


prompt = PromptTemplate(template=template, input_variables=['context', 'question'], output_parser=ScoreOutputParser())
