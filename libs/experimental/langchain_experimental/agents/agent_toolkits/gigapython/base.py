from typing import List, Optional, Tuple

from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_experimental.agents.agent_toolkits.gigapython.parser import (
    CodeOutputParser,
)

SYSTEM_PROMPT = """Ты — агент, предназначенный для помощи пользователю и написания Python кода для ответа на вопросы. 
Если тебя просят нарисовать график, то напиши python код рисующий график с помощью matplotlib. 
Если в результате выполнения кода произошла ошибка, исправь код и напиши обновленный код
Используй только результат выполнения кода для ответа на вопрос пользователя"""  # noqa: E501

code_prompt = PromptTemplate.from_template(
    """{system}

{input}
{agent_scratchpad}"""  # noqa
)

code_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system}"),
        MessagesPlaceholder("history", optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    messages = ""
    for action, observation in intermediate_steps[:]:
        messages += f"""{action.log}\n{observation}\n"""
    return messages


def format_log_to_messages(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Construct the scratchpad that lets the agent continue its thought process."""
    messages = []
    for action, observation in intermediate_steps[:]:
        messages += [AIMessage(content=action.log), HumanMessage(content=observation)]
    return messages


def create_code_agent(
    llm: BaseLanguageModel, prompt: Optional[BasePromptTemplate] = None
) -> Runnable:
    """
    Создаем агента, который выполняет python код. Подходит для классов LLM и ChatModel
    """
    if prompt is None:
        prompt = code_prompt
    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"])
        )
        | prompt.partial(system=SYSTEM_PROMPT)
        | llm
        | CodeOutputParser()
    )


def create_code_chat_agent(
    llm: BaseLanguageModel[BaseMessage], prompt: Optional[BasePromptTemplate] = None
) -> Runnable:
    """
    Создаем агента, который выполняет python код. Подходит для классов ChatModel
    """
    if prompt is None:
        prompt = code_chat_prompt
    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(x["intermediate_steps"])
        )
        | prompt.partial(system=SYSTEM_PROMPT)
        | llm
        | CodeOutputParser()
    )
