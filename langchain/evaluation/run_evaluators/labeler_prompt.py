# flake8: noqa

from langchain.prompts import PromptTemplate

template = """You are labeling a submitted answer on a given task or input based on a set of labels. Here is the data:
[BEGIN DATA]
***
[Task]: {input}
***
[Submission]: {output}
***
[Labels]: {labels}
***
[END DATA]
Please analyze the submission carefully considering the task it was supposed to accomplish. Compare it with the provided labels. Your task is to choose the most fitting label for the submission. Avoid simply stating the correct label at the outset. Write out in a step by step manner your reasoning about the label choice to be sure that your conclusion is correct. At the end, print the label that you believe is most appropriate for the submission on its own line. Repeat the label again by itself on a new line."""

PROMPT = PromptTemplate(
    input_variables=["input", "output", "labels"], template=template
)
