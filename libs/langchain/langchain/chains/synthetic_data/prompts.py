from langchain.prompts.prompt import PromptTemplate

sentence_template = """Given the following fields, create a sentence about them. Make it detailed and interesting.

Fields:
{fields}
Sentence:
"""

SENTENCE_PROMPT = PromptTemplate(template=sentence_template, input_variables=["fields"])
