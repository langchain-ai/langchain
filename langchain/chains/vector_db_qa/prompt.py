# flake8: noqa
from typing import Any, List

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import ChatMessage

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


chat_template_system = """Use the following pieces of context to answer any user questions. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}"""
chat_prompt = PromptTemplate.from_template(chat_template_system)
question_prompt = PromptTemplate.from_template("{question}")

CHAT_PROMPT = ChatPromptTemplate(
    messages=[("system", chat_prompt), ("user", question_prompt)],
    input_variables=["context", "question"],
)
