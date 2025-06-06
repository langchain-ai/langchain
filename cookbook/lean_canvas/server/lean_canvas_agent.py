from typing import Literal, Optional, TypeAlias

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langchain_gigachat import GigaChat
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

load_dotenv(find_dotenv())

llm = GigaChat(model="GigaChat-2-Max", profanity_check=False, top_p=0, timeout=120)


class LeanGraphState(TypedDict):
    main_task: Annotated[str, "Основная задача от пользователя"]
    competitors_analysis: Optional[Annotated[str, "Анализ конкурентов"]]
    feedback: Optional[
        Annotated[
            str, "Фидбек от пользователя. Обязательно учитывай его в своих ответах!"
        ]
    ]

    # Lean Canvas
    problem: Optional[
        Annotated[str, "Проблема, которую пытается решить продукт или услуга."]
    ]
    solution: Optional[Annotated[str, "Краткое описание предлагаемого решения."]]
    key_metrics: Optional[
        Annotated[
            str,
            "Ключевые показатели, которые необходимо измерять для отслеживания прогресса.",
        ]
    ]
    unique_value_proposition: Optional[
        Annotated[
            str,
            "Единое, ясное и убедительное сообщение, объясняющее, почему вы отличаетесь от других и почему стоит покупать именно у вас.",
        ]
    ]
    unfair_advantage: Optional[
        Annotated[str, "То, что конкуренты не могут легко скопировать или купить."]
    ]
    channels: Optional[Annotated[str, "Пути охвата ваших клиентских сегментов."]]
    customer_segments: Optional[
        Annotated[
            str, "Целевая аудитория или группы людей, которых вы пытаетесь охватить."
        ]
    ]
    cost_structure: Optional[
        Annotated[str, "Основные затраты, связанные с ведением бизнеса."]
    ]
    revenue_streams: Optional[Annotated[str, "Как бизнес будет зарабатывать деньги."]]


def state_to_string(state: LeanGraphState) -> str:
    """
    Преобразует состояние в строку для отображения.
    """
    result = []
    for field, annotation in LeanGraphState.__annotations__.items():
        value = state.get(field, "")
        if value:
            # annotation is typing.Annotated[type, description]
            if hasattr(annotation, "__metadata__") and annotation.__metadata__:
                desc = annotation.__metadata__[0]
            else:
                desc = ""
            result.append(f"{desc} ({field}): {value}")
    return "\n".join(result)


def ask_llm(state: LeanGraphState, question: str, config: RunnableConfig) -> str:
    TEMPLATE = """
    Ты - эксперт в области стартапов и Lean Canvas. Твоя задача - помочь пользователю создать Lean Canvas для его задачи.
    Учитывай уже заполненные части таблицы Lean Canvas и главную задачу пользователя (main_task).
    
    Обязательно учитывай фидбек от пользователя (feedback), если он задан.
    <STATE>
    {state}
    </STATE>
    
    Ответь на вопрос: {question}
    Отвечай коротко, не более 1-2 коротких предложений и обязательно учти фидбек от пользователя (feedback), если он задан. Оформи ответ в виде буллетов.
    """

    prompt = ChatPromptTemplate.from_messages([("system", TEMPLATE)])

    model = config["configurable"].get("model", "GigaChat-2-Max")
    chain = prompt | llm.bind(model=model) | StrOutputParser()
    return chain.invoke({"state": state_to_string(state), "question": question})


def customer_segments(state: LeanGraphState, config: RunnableConfig):
    return {"customer_segments": ask_llm(state, "Кто ваши целевые клиенты?", config)}


def problem(state: LeanGraphState, config: RunnableConfig):
    return {"problem": ask_llm(state, "Какую проблему вы решаете?", config)}


def unique_value_proposition(state: LeanGraphState, config: RunnableConfig):
    return {
        "unique_value_proposition": ask_llm(
            state, "Какое уникальное предложение вы предлагаете?", config
        )
    }


def solution(state: LeanGraphState, config: RunnableConfig):
    return {
        "solution": ask_llm(
            state, "Какое решение вы предлагаете для этой проблемы?", config
        )
    }


def channels(state: LeanGraphState, config: RunnableConfig):
    return {
        "channels": ask_llm(
            state, "Какие каналы привлечения клиентов вы используете?", config
        )
    }


def revenue_streams(state: LeanGraphState, config: RunnableConfig):
    return {
        "revenue_streams": ask_llm(
            state, "Как вы планируете зарабатывать деньги?", config
        )
    }


def cost_structure(state: LeanGraphState, config: RunnableConfig):
    return {"cost_structure": ask_llm(state, "Какова структура ваших затрат?", config)}


def key_metrics(state: LeanGraphState, config: RunnableConfig):
    return {
        "key_metrics": ask_llm(
            state, "Какие ключевые показатели вы будете отслеживать?", config
        )
    }


def unfair_advantage(state: LeanGraphState, config: RunnableConfig):
    return {
        "unfair_advantage": ask_llm(
            state, "Какое ваше конкурентное преимущество?", config
        )
    }


class CompetitorsAnalysisResult(BaseModel):
    """Анализ конкурентов"""

    thoughts: str = Field(description="Мысли по поводу ответа")
    solution: str = Field(
        description="Какие конкуренты существуют и чем они отличаются от вашего продукта"
    )
    is_unique: bool = Field(description="Уникально ли ваше предложение?")


COMPETITION_ANALYSIS_TEMPLATE = """Ты работаешь над таблицей Lean Canvas и тебе нужно проанализировать конкурентов.

Учитывай уже заполненные части таблицы Lean Canvas и главную задачу пользователя (main_task).
<STATE>
{state}
</STATE>

Результаты поиска по запросу "{unique_value_proposition}". Учитывай их, чтобы понять, уникальную ли идею ты придумал.
Если в поиске нет ничего похожего, значит идея вероятно уникальная.
<SEARCH_RESULTS>
{search_results}
</SEARCH_RESULTS>

Выведи только следующую информацию в формате JSON:
{format_instructions}"""


def check_unique(
    state: LeanGraphState, config: RunnableConfig
) -> Command[Literal["4_solution", "3_unique_value_proposition"]]:
    if config["configurable"].get("skip_search", False):
        # Если пропускаем поиск, то просто переходим к следующему шагу
        return Command(goto="4_solution")

    parser = PydanticOutputParser(pydantic_object=CompetitorsAnalysisResult)
    prompt = ChatPromptTemplate.from_messages(
        [("system", COMPETITION_ANALYSIS_TEMPLATE)]
    ).partial(format_instructions=parser.get_format_instructions())

    search_results_text = TavilySearch().run(state["unique_value_proposition"])

    chain = prompt | llm | parser
    res = chain.invoke(
        {
            "state": state_to_string(state),
            "unique_value_proposition": state["unique_value_proposition"],
            "search_results": search_results_text,
        }
    )

    competitors_analysis = (
        state.get("competitors_analysis", "")
        + "\n"
        + state["unique_value_proposition"]
        + " - "
        + res.solution
    )

    if res.is_unique:
        # Если предложение уникально, переходим к следующему шагу
        return Command(
            update={"competitors_analysis": competitors_analysis.strip()},
            goto="4_solution",
        )
    else:
        # Если предложение не уникально, возвращаемся к шагу "3_unique_value_proposition"
        return Command(
            update={"competitors_analysis": competitors_analysis.strip()},
            goto="3_unique_value_proposition",
        )


RedirectStep: TypeAlias = Literal[
    "1_customer_segments",
    "2_problem",
    "3_unique_value_proposition",
    "4_solution",
    "5_channels",
    "6_revenue_streams",
    "7_cost_structure",
    "8_key_metrics",
    "9_unfair_advantage",
    "__end__",
]


class UserFeedback(BaseModel):
    """Анализ конкурентов"""

    feedback: str = Field(description="Фидебек пользователя, что надо исправить")
    next_step: RedirectStep = Field(description="Следующий шаг в Lean Canvas")
    is_done: bool = Field(description="Можно ли завершать создание Lean Canvas?")


FEEDBACK_TEMPLATE = """Ты работаешь над таблицей Lean Canvas. Ты уже сгенерировал версию Lean Canvas и получил фидбек от пользователя.
Тебе нужно разобраться фидбек и понять, как действовать дальше, заполнив таблицу с ответом.

Учитывай уже заполненные части таблицы Lean Canvas и главную задачу пользователя (main_task).
<STATE>
{state}
</STATE>

Вот фидбек пользователя на твою работу:
{feedback}

Извлеки из него данные для дальнейшей работы. Если пользователь всем доволен или не говорит ничего конкретного, 
то прими решение закончить генерацию (is_done = True).
Выведи только следующую информацию в формате JSON:
{format_instructions}"""


def get_feedback(
    state: LeanGraphState, config: RunnableConfig
) -> Command[Literal[RedirectStep, END]]:
    if config["configurable"].get("need_interrupt"):
        feedback = interrupt("""Пожалуйста, дайте обратную связь по Lean Canvas. Если все хорошо, напишите 'Хорошо'. 
    Если нужно что-то изменить, напишите, что именно и с какого шага начать.""")
    else:
        feedback = "Все хорошо!"

    parser = PydanticOutputParser(pydantic_object=UserFeedback)
    prompt = ChatPromptTemplate.from_messages([("system", FEEDBACK_TEMPLATE)]).partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | llm | parser
    res = chain.invoke(
        {
            "state": state_to_string(state),
            "feedback": feedback,
        }
    )

    if res.is_done:
        return Command(update={}, goto=END)
    else:
        # Если предложение не уникально, возвращаемся к шагу "3_unique_value_proposition"
        return Command(
            update={"feedback": res.feedback},
            goto=res.next_step,
        )


class GraphConfig(BaseModel):
    need_interrupt: bool = True


graph = StateGraph(LeanGraphState, config_schema=GraphConfig)

graph.add_node("1_customer_segments", customer_segments)
graph.add_node("2_problem", problem)
graph.add_node("3_unique_value_proposition", unique_value_proposition)
graph.add_node("3.1_check_unique", check_unique)
graph.add_node("4_solution", solution)
graph.add_node("5_channels", channels)
graph.add_node("6_revenue_streams", revenue_streams)
graph.add_node("7_cost_structure", cost_structure)
graph.add_node("8_key_metrics", key_metrics)
graph.add_node("9_unfair_advantage", unfair_advantage)
graph.add_node("get_feedback", get_feedback)

graph.add_edge(START, "1_customer_segments")
graph.add_edge("1_customer_segments", "2_problem")
graph.add_edge("2_problem", "3_unique_value_proposition")
graph.add_edge("3_unique_value_proposition", "3.1_check_unique")
graph.add_edge("4_solution", "5_channels")
graph.add_edge("5_channels", "6_revenue_streams")
graph.add_edge("6_revenue_streams", "7_cost_structure")
graph.add_edge("7_cost_structure", "8_key_metrics")
graph.add_edge("8_key_metrics", "9_unfair_advantage")
graph.add_edge("9_unfair_advantage", "get_feedback")

agent = graph.compile()
