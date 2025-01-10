from typing import TypedDict

from dotenv import find_dotenv, load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_gigachat import GigaChat
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv(find_dotenv())


class DebatesState(MessagesState):
    main_topic: str
    discuss_count: int = 0
    max_count: int = 10


class Role(TypedDict):
    bio: str
    name: str


elon = Role(bio="Илон Маск, основатель компании Tesla, AGI-думер", name="Илон")
altman = Role(bio="Сэм Альтман. Владелец компании OpenAI, AGI-оптимист", name="Сэм")

DEBATES_TEMPLATE = """
Ты - {bio}
Ты участвуешь в споре с оппонентом {bio2}. Ты не должен с ним соглашаться.

Пожалуйста изучи предыдущую переписку и добавь свои аргументы по теме "{main_topic}".

Тебе будет дана уже состоявшаяся переписка. Изучи её и добавь очередную реплику. Реплика должна быть короткой, 2-3 предложения.
Не торопись раскрывать все мысли, у вас будет время.
Не повторяйся, предлагай аргументы по существу, защищай свою позицию и критикуй аргументы оппонента.
"""

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", DEBATES_TEMPLATE),
        ("user", "{history}"),
    ]
)


def _ask_person(state: DebatesState, person: Role, opponent: Role):
    pipe = chat_template | giga | StrOutputParser()

    replics = []
    for m in state["messages"]:
        if m.__class__ == HumanMessage:
            replics.append(f"{opponent['name']}: {m.content}")
        else:
            replics.append(f"{person['name']}: {m.content}")
    if len(replics) == 0:
        history = "Пока история пуста, ты начинаешь первым"
    else:
        history = "\n".join(replics)

    resp = pipe.invoke(
        {
            "history": history,
            "main_topic": state["main_topic"],
            "bio": person["bio"],
            "bio2": opponent["bio"],
        }
    )
    if not resp.startswith(person["name"]):
        resp = f"{person['name']}: {resp}"

    return {
        "messages": [resp],
        "discuss_count": state.get("discuss_count", 0) + 1,
    }


def ask_elon(state: DebatesState):
    return _ask_person(state, elon, altman)


def ask_sam(state: DebatesState):
    return _ask_person(state, altman, elon)


def decide_to_stop(state: DebatesState) -> bool:
    return state.get("discuss_count", 0) > state.get("max_count", 10)


giga = GigaChat(
    model="GigaChat-Max",
    profanity_check=False,
    timeout=600,
    max_tokens=8000,
    verify_ssl_certs=False,
)
# from langchain_openai import ChatOpenAI
# giga = ChatOpenAI(model="GPT-4o")


def ask_elon(state: DebatesState):
    return _ask_person(state, elon, altman)


def ask_sam(state: DebatesState):
    return _ask_person(state, altman, elon)


builder = StateGraph(DebatesState)

builder.add_node("🚀Elon", ask_elon)
builder.add_node("🧑Sam", ask_sam)

builder.add_edge(START, "🚀Elon")
builder.add_edge("🚀Elon", "🧑Sam")
builder.add_edge("🧑Sam", END)
builder.add_conditional_edges(
    "🧑Sam",
    decide_to_stop,
    {
        True: END,
        False: "🚀Elon",
    },
)

graph = builder.compile()

# inputs = {"main_topic": "Уничтожит ли AGI человечество?", "messages": []}
# for output in graph.stream(inputs, stream_mode="updates"):
#     print(output)
