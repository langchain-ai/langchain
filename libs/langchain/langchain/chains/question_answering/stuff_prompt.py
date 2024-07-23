# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt_template = """Используй информацию между тэгами BEGIN_CONTEXT и END_CONTEXT, чтобы ответить на вопросы пользователя.Если ты не знаешь ответа и в данной тебе информации между тэгами BEGIN_CONTEXT и END_CONTEXT ее нет, просто скажи, что не знаешь, не пытайся придумать ответ.

BEGIN_CONTEXT
{context}
END_CONTEXT

Question: {question}
Полезный ответ:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """Используй информацию между тэгами BEGIN_CONTEXT и END_CONTEXT, чтобы ответить на вопросы пользователя.Если ты не знаешь ответа и в данной тебе информации между тэгами BEGIN_CONTEXT и END_CONTEXT ее нет, просто скажи, что не знаешь, не пытайся придумать ответ.

BEGIN_CONTEXT
{context}
END_CONTEXT"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
