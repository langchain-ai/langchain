import random

from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent, ZeroShotAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI, GigaChat
from langchain.tools import BaseTool, StructuredTool, Tool, tool

llm = GigaChat()

PRODUCT_DATABASE = {
    'творог': 100,
    'сыр': 200,
    'пиво': 90
}


def search_product(name: str) -> str:
    name = name.replace('"', '').replace("'", '').lower()
    return f"{PRODUCT_DATABASE[name]} руб."


songs = [
    {
        'name': 'Грустная песня',
        'cf': 0
    },
    {
        'name': 'Почти грустная песня, но не совсем',
        'cf': 15
    },
    {
        'name': 'Ну нормальная песня не радостная не веселая середнячок норм',
        'cf': 50
    },
    {
        'name': 'Почти счастливая песня',
        'cf': 75
    },
    {
        'name': 'Счастливая песня',
        'cf': 100
    }
]


def create_playlist(cf):
    cf = int(cf)
    result = ",\n".join(
        [song['name'] for song in songs if cf - 30 <= song['cf'] <= cf + 30])
    return f'Cоздан следующий плейлист "Название плейлиста":\n{result}'


def search_info(query):
    print(query)
    return 'Информация: В России проживает 12 человек, а также в России есть и тропинка и лесок и речка и небо надо мною'


tools = [
    Tool.from_function(
        func=create_playlist,
        name="CREATE_PLAYLIST",
        description="Описание: создает плейлисты \n Параметры: число которое обозначает настроение запроса где 0 - грусть, печаль и 100 - счастье, веселье\n"
        # noqa: E501
        # noqa: E501
    ),
    Tool.from_function(
        func=lambda query: "10 руб.",
        name="SEARCH",
        description="Описание: находит цену продукта в базе данных \n Параметры: строка с названием продукта\n"
        # noqa: E501
    ),
    Tool.from_function(
        func=lambda query: '1$',
        name="CONVERT_CURRENCY",
        description="Описание: переводит цену в рублях в доллары \n Параметры: строка с ценой\n"
        # noqa: E501
    ),
]

PREFIX = """
Ты бот-ассистент. Ты можешь выбирать какую функцию выполнить и какие параметры ей передать.
Ты ничего не знаешь о ценах на товары и не знаешь как переводить цены в валюту
Ты не умеешь выполнять функции и получать ответы функций. Их выполняет пользователь
Ответь не следующие вопросы как можно лучше. Ты можешь вызвать следующие функции:"""

agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prefix=PREFIX)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

print(agent_executor.run(
    "Найди сколько стоит творог и переведи его цену в доллары"
))
