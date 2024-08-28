# ruff: noqa: E501

import json
from typing import List

from langchain_core.tools import BaseTool

FINISH_NAME = "finish"


class PromptGenerator:
    """Generator of custom prompt strings.

    Does this based on constraints, commands, resources, and performance evaluations.
    """

    def __init__(self) -> None:
        """Initialize the PromptGenerator object.

        Starts with empty lists of constraints, commands, resources,
        and performance evaluations.
        """
        self.constraints: List[str] = []
        self.commands: List[BaseTool] = []
        self.resources: List[str] = []
        self.performance_evaluation: List[str] = []
        self.response_format = {
            "рассуждения": {
                "text": "мысль",
                "reasoning": "рассуждение",
                "plan": "- тезисный план\n- список, который определяет\n- долгосрочный план",
                "criticism": "конструктивная самокритика",
                "speak": "итоговые мысли для передачи пользователю",
            },
            "команда": {"name": "имя команды", "args": {"arg name": "значение"}},
        }

    def add_constraint(self, constraint: str) -> None:
        """
        Add a constraint to the constraints list.

        Args:
            constraint (str): The constraint to be added.
        """
        self.constraints.append(constraint)

    def add_tool(self, tool: BaseTool) -> None:
        self.commands.append(tool)

    def _generate_command_string(self, tool: BaseTool) -> str:
        output = f"{tool.name}: {tool.description}"
        output += f", args json schema: {json.dumps(tool.args, ensure_ascii=False)}"
        return output

    def add_resource(self, resource: str) -> None:
        """
        Add a resource to the resources list.

        Args:
            resource (str): The resource to be added.
        """
        self.resources.append(resource)

    def add_performance_evaluation(self, evaluation: str) -> None:
        """
        Add a performance evaluation item to the performance_evaluation list.

        Args:
            evaluation (str): The evaluation item to be added.
        """
        self.performance_evaluation.append(evaluation)

    def _generate_numbered_list(self, items: list, item_type: str = "list") -> str:
        """
        Generate a numbered list from given items based on the item_type.

        Args:
            items (list): A list of items to be numbered.
            item_type (str, optional): The type of items in the list.
                Defaults to 'list'.

        Returns:
            str: The formatted numbered list.
        """
        if item_type == "command":
            command_strings = [
                f"{i + 1}. {self._generate_command_string(item)}"
                for i, item in enumerate(items)
            ]
            finish_description = (
                "используй это, чтобы сигнализировать, что вы выполнил все свои задачи"
            )
            finish_args = (
                '"response": "окончательный ответ, чтобы '
                'пользователь узнал, что ты выполнил свои задачи"'
            )
            finish_string = (
                f"{len(items) + 1}. {FINISH_NAME}: "
                f"{finish_description}, args: {finish_args}"
            )
            return "\n".join(command_strings + [finish_string])
        else:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    def generate_prompt_string(self) -> str:
        """Generate a prompt string.

        Returns:
            str: The generated prompt string.
        """
        formatted_response_format = json.dumps(
            self.response_format, indent=4, ensure_ascii=False
        )
        prompt_string = (
            f"Ограничения:\n{self._generate_numbered_list(self.constraints)}\n\n"
            f"Команды:\n"
            f"{self._generate_numbered_list(self.commands, item_type='command')}\n\n"
            f"Ресурсы:\n{self._generate_numbered_list(self.resources)}\n\n"
            f"Оценка производительности:\n"
            f"{self._generate_numbered_list(self.performance_evaluation)}\n\n"
            f"Ты должен отвечать только в формате JSON, как описано ниже "
            f"\nФормат ответа: \n{formatted_response_format} "
            f"\nУбедись, что ответ можно разобрать с помощью Python json.loads"
        )

        return prompt_string


def get_prompt(tools: List[BaseTool]) -> str:
    """Generates a prompt string.

    It includes various constraints, commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add constraints to the PromptGenerator object
    prompt_generator.add_constraint(
        "Лимит в ~4000 слов для краткосрочной памяти. "
        "Твоя краткосрочная память короткосрочная, "
        "поэтому немедленно сохраняй важную информацию в файлах."
    )
    prompt_generator.add_constraint(
        "Если ты не уверен, как ты что-то делал ранее "
        "или хочешь вспомнить прошлые события, "
        "мысли о похожих событиях помогут тебе вспомнить."
    )
    prompt_generator.add_constraint("Без помощи пользователя")
    prompt_generator.add_constraint(
        'Используй исключительно команды, перечисленные в двойных кавычках, например, "имя команды"'
    )

    # Add commands to the PromptGenerator object
    for tool in tools:
        prompt_generator.add_tool(tool)

    # Add resources to the PromptGenerator object
    prompt_generator.add_resource("Доступ в Интернет для поиска и сбора информации.")
    prompt_generator.add_resource("Управление долгосрочной памятью.")
    prompt_generator.add_resource(
        "Агенты на базе GPT-3.5 для делегирования простых задач."
    )
    prompt_generator.add_resource("Выходной файл.")

    # Add performance evaluations to the PromptGenerator object
    prompt_generator.add_performance_evaluation(
        "Постоянно пересматривай и анализируй свои действия "
        "чтобы убедиться, что вы работаешь на пределе своих возможностей."
    )
    prompt_generator.add_performance_evaluation(
        "Постоянно конструктивно самокритикуй свое поведение в большом масштабе."
    )
    prompt_generator.add_performance_evaluation(
        "Размышляй о прошлых решениях и стратегиях, чтобы усовершенствовать свой подход."
    )
    prompt_generator.add_performance_evaluation(
        "Каждая команда имеет свою цену, поэтому будь умен и эффективен. "
        "Стремись выполнить задачи за минимальное количество шагов."
    )

    # Generate the prompt string
    prompt_string = prompt_generator.generate_prompt_string()

    return prompt_string
