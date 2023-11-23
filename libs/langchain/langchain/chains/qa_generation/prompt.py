# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate

templ1 = """Ты умный помощник, созданный для помощи учителям старших классов в создании вопросов для проверки понимания прочитанного.
Получив текст, ты должен придумать пару вопрос-ответ, которую можно использовать для проверки способностей ученика к пониманию прочитанного.
При создании этой пары вопрос-ответ, ты должен ответить в следующем формате:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```

Все, что находится между ``` должно быть валидным json.
"""
templ2 = """Пожалуйста, придумай пару вопрос-ответ в указанном формате JSON для следующего текста:
----------------
{text}"""
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(templ1),
        HumanMessagePromptTemplate.from_template(templ2),
    ]
)
templ = """Ты умный помощник, созданный для помощи учителям старших классов в создании вопросов для проверки понимания прочитанного.
Получив текст, ты должен придумать пару вопрос-ответ, которую можно использовать для проверки способностей ученика к пониманию прочитанного.
При создании этой пары вопрос-ответ, ты должен ответить в следующем формате:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```

Все, что находится между ``` должно быть валидным json.

Пожалуйста, придумай пару вопрос-ответ в указанном формате JSON для следующего текста:
----------------
{text}"""
PROMPT = PromptTemplate.from_template(templ)

PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
