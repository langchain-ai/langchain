# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate

question_prompt_template = """Используй следующий фрагмент длинного документа, чтобы увидеть, содержит ли текст информацию, относящуюся к ответу на вопрос. 
Верни любой релевантный текст дословно.
{context}
Question: {question}
Релевантный текст, если таковой имеется:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)
system_template = """Используй следующий фрагмент длинного документа, чтобы увидеть, содержит ли текст информацию, относящуюся к ответу на вопрос. 
Верни любой релевантный текст дословно.
______________________
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(messages)


QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=QUESTION_PROMPT, conditionals=[(is_chat_model, CHAT_QUESTION_PROMPT)]
)

combine_prompt_template = """Исходя из следующих выделенных частей длинного документа и вопроса, сформулируй окончательный ответ. 
Если ты не знаешь ответа, просто скажи, что не знаешь. Не пытайся выдумать ответ.

Question: Какое государство/страна регулирует толкование контракта?
=========
Содержание: Это Соглашение регулируется английским законодательством, и стороны подчиняются исключительной юрисдикции английских судов в отношении любого спора (контрактного или внеконтрактного) по данному Соглашению, за исключением того, что любая из сторон может обратиться в любой суд за получением судебного запрета или иного средства защиты своих прав интеллектуальной собственности.

Содержание: Нет отказа. Невыполнение или задержка в осуществлении любого права или средства правовой защиты по данному Соглашению не составляет отказа от такого (или любого другого) права или средства правовой защиты.\n\n11.7 Разделимость. Недействительность, незаконность или неосуществимость любого условия (или его части) данного Соглашения не влияет на продолжение действия остальной части условия (если таковая имеется) и данного Соглашения.\n\n11.8 Нет агентства. За исключением как это прямо указано иначе, ничто в данном Соглашении не создает агентства, партнерства или совместного предприятия любого рода между сторонами.\n\n11.9 Нет третьих лиц-бенефициаров.

Содержание: (b) если Google верит, в доброй вере, что Дистрибьютор нарушил или заставил Google нарушить любые Антикоррупционные законы (как определено в пункте 8.5) или что такое нарушение вполне вероятно,
=========
FINAL ANSWER: Это Соглашение регулируется английским законодательством.

Question: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

system_template = """Исходя из следующих выделенных частей длинного документа и вопроса, сформулируй окончательный ответ. 
Если ты не знаешь ответа, просто скажи, что не знаешь. Не пытайся выдумать ответ.
______________________
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_COMBINE_PROMPT = ChatPromptTemplate.from_messages(messages)


COMBINE_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=COMBINE_PROMPT, conditionals=[(is_chat_model, CHAT_COMBINE_PROMPT)]
)
