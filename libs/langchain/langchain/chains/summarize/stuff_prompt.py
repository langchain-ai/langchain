from langchain_core.prompts import PromptTemplate

prompt_template = """Write a concise summary of the following:


"{text}"


CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
