from langchain_core.prompts import PromptTemplate

llm_context_prompt_template = """
Answer the user query using provided passages. Each passage has metadata given as 
a nested JSON object you can also use. When answering, cite source name of the passages 
you are answering from below the answer in a unique bullet point list.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

----
{context}
----
Query: {query}
"""  # noqa: E501

LLM_CONTEXT_PROMPT = PromptTemplate.from_template(llm_context_prompt_template)
