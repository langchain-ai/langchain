# ruff: noqa: E501

import json
from textwrap import dedent
from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from langchain_experimental.tot.thought import ThoughtValidity


def get_cot_prompt() -> PromptTemplate:
    """Get the prompt for the Chain of Thought (CoT) chain."""

    return PromptTemplate(
        template_format="jinja2",
        input_variables=["problem_description", "thoughts"],
        template=dedent(
            """
            Ты - интеллектуальный агент, который генерирует одну мысль за раз в
            древовидной структуре мыслей.

            ПРОБЛЕМА 
            
            {{problem_description}}
            
            {% if thoughts %}
            МЫСЛИ
            
            {% for thought in thoughts %}
            {{ thought }}
            {% endfor %}
            {% endif %}
            
            Давайте думать шаг за шагом.
            """
        ).strip(),
    )


class JSONListOutputParser(BaseOutputParser):
    """Parse the output of a PROPOSE_PROMPT response."""

    @property
    def _type(self) -> str:
        return "json_list"

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""

        json_string = text.split("```json")[1].strip().strip("```").strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return []


def get_propose_prompt() -> PromptTemplate:
    """Get the prompt for the PROPOSE_PROMPT chain."""

    return PromptTemplate(
        template_format="jinja2",
        input_variables=["problem_description", "thoughts", "n"],
        output_parser=JSONListOutputParser(),
        template=dedent(
            """
            Ты - интеллектуальный агент, который генерирует мысли в древовидной
            структуре мыслей.

            Вывод должен быть оформлен в виде фрагмента кода на markdown, отформатированного как JSON-список
            строк, включая ведущие и замыкающие "```json" и "```":

            ```json
            [
                "<мысль-1>",
                "<мысль-2>",
                "<мысль-3>"
            ]
            ```

            ПРОБЛЕМА

            {{ problem_description }}

            {% if thoughts %}
            ВАЛИДНЫЕ МЫСЛИ

            {% for thought in thoughts %}
            {{ thought }}
            {% endfor %}

            Возможные следующие {{ n }} валидные мысли на основе последней валидной мысли:
            {% else %}

            Возможные следующие {{ n }} валидные мысли на основе ПРОБЛЕМЫ:
            {%- endif -%}
        """
        ).strip(),
    )


class CheckerOutputParser(BaseOutputParser):
    """Parse and check the output of the language model."""

    def parse(self, text: str) -> ThoughtValidity:
        """Parse the output of the language model."""
        text = text.upper()
        if "INVALID" in text:
            return ThoughtValidity.INVALID
        elif "INTERMEDIATE" in text:
            return ThoughtValidity.VALID_INTERMEDIATE
        elif "VALID" in text:
            return ThoughtValidity.VALID_FINAL
        else:
            return ThoughtValidity.INVALID

    @property
    def _type(self) -> str:
        return "tot_llm_checker_output"


CHECKER_PROMPT = PromptTemplate(
    input_variables=["problem_description", "thoughts"],
    template=dedent(
        """
        Ты - интеллектуальный агент, проверяющий мысли другого интеллектуального агента.

        ПРОБЛЕМА 
        
        {problem_description}

        МЫСЛИ
        
        {thoughts}

        Оцени мысли и ответь одним словом.

        - Ответь ВАЛИДНО, если последняя мысль является валидным окончательным решением проблемы.
        - Ответь НЕВАЛИДНО, если последняя мысль невалидна.
        - Ответь ПРОМЕЖУТОЧНО, если последняя мысль валидна, но не является окончательным 
        решением проблемы.

        Эта цепочка мыслей"""
    ).strip(),
    output_parser=CheckerOutputParser(),
)
