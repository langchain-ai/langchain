# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

combined_prompt_emplate = """Given the following chat history and pieces of contexts, please answer the follow up question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
---
Chat History: {chat_history}
---
Context: {context}
---
Question: {question}
Helpful Answer:"""
CHAT_RETRIEVAL_QA_PROMPT = PromptTemplate(
    template=combined_prompt_emplate,
    input_variables=["chat_history", "context", "question"],
)
