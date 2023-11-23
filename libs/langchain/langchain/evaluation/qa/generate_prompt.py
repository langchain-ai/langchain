# flake8: noqa
from langchain.output_parsers.regex import RegexParser
from langchain_core.prompts import PromptTemplate

template = """Ты учитель, который составляет вопросы для викторины. 
Исходя из следующего документа, пожалуйста, сформулируй вопрос и ответ, основанные на этом документе.

Пример формата:
<Начало документа>
...
<Конец документа>
ВОПРОС: вопрос здесь
ОТВЕТ: ответ здесь

Эти вопросы должны быть подробными и строго основываться на информации в документе. Начни!

<Начало документа>
{doc}
<Конец документа>"""
PROMPT = PromptTemplate(
    input_variables=["doc"],
    template=template,
)
