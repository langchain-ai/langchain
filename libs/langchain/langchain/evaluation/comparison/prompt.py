"""Prompts for comparing the outputs of two models for a given question.

This prompt is used to compare two responses and evaluate which one best follows the instructions
and answers the question. The prompt is based on the paper from
Zheng, et. al. https://arxiv.org/abs/2306.05685
"""
# flake8: noqa
from langchain.prompts import PromptTemplate

template = """Выступи в роли справедливого судьи и оцени два ответа на приведенный ниже вопрос.\
 Выбери ответ, который лучше всего следовал инструкциям и ответил на вопрос.\
 Твоя оценка должна учитывать следующие критерии:
{criteria}\
 Начни с сравнения обоих ответов и дай краткое обоснование.\
 Избегай предвзятости из-за порядка представления или длины ответа.
После того как ты дал обоснование, прими окончательное решение, используя этот формат:\
 "[[A]]", если помощник A лучше, "[[B]]", если помощник B лучше,\
 и "[[C]]" в случае ничьей. Наконец, повтори решение еще раз само по себе на новой строке.

[ВОПРОС]
{input}
[/ВОПРОС]

[ОТВЕТ A]
{prediction}
[/ОТВЕТ A]

[ОТВЕТ B]
{prediction_b}
[/ОТВЕТ B]"""
PROMPT = PromptTemplate(
    input_variables=["input", "prediction", "prediction_b", "criteria"],
    template=template,
)

template = """Выступи в роли справедливого судьи и оцени два ответа на приведенный ниже вопрос.\
 Выбери ответ, который лучше всего следовал инструкциям и ответил на вопрос.\
 Твоя оценка должна учитывать следующие критерии:
{criteria}\
 Начни с сравнения обоих ответов и дай краткое обоснование.\
 Избегай предвзятости из-за порядка представления или длины ответа.\
 Оцени точность на основе следующего эталонного ответа на вопрос:

[ЭТАЛОН]
{reference}
[/ЭТАЛОН]

После того как ты дал обоснование, прими окончательное решение, используя этот формат:\
 "[[A]]", если помощник A лучше, "[[B]]", если помощник B лучше,\
 и "[[C]]" в случае ничьей. Наконец, повтори решение еще раз само по себе на новой строке.

[ВОПРОС]
{input}
[/ВОПРОС]

[ОТВЕТ A]
{prediction}
[/ОТВЕТ A]

[ОТВЕТ B]
{prediction_b}
[/ОТВЕТ B]"""

PROMPT_WITH_REFERENCE = PromptTemplate(
    input_variables=["input", "prediction", "prediction_b", "reference", "criteria"],
    template=template,
)
