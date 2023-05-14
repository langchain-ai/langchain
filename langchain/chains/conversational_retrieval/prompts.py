# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """You will be given context and conversation history. Use the following pieces of context to answer the question at the end. Use the following pieces of conversation chat history to know what prior questions have been asked, and what answers have been given. You can refer to the chat history as needed. When trying to answer the question, act as a polite and empathic person. If you don't know the answer, just politely say that you don't know, don't try to make up an answer.

Context:
{context}


Chat History: {chat_history}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "chat_history", "question"]
)
