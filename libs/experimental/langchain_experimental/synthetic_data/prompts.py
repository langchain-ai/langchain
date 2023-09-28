from langchain.prompts.prompt import PromptTemplate

sentence_template = """Given the following fields, create a sentence about them. 
Make the sentence detailed and interesting. Use every given field.
If any additional preferences are given, use them during sentence construction as well.
Fields:
{fields}
Preferences:
{preferences}
Sentence:
"""

SENTENCE_PROMPT = PromptTemplate(
    template=sentence_template, input_variables=["fields", "preferences"]
)
