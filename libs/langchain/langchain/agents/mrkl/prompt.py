# flake8: noqa
PREFIX = """Ответь на следующие вопросы как можно лучше. У тебя есть следующие инструменты:"""
FORMAT_INSTRUCTIONS = """Используй следующий формат:

Question: входной вопрос, на который ты должен ответить
Thought: ты всегда должен думать о том, что делать
Action: действие, которое следует предпринять, должно быть одним из [{tool_names}]
Action Input: ввод для действия
Observation: результат действия
... (этот циклThought/Action/Action Input/Observation может повторяться N раз)
Thought: Теперь я знаю окончательный ответ
Final answer: окончательный ответ на исходный вопрос"""
SUFFIX = """Начни!

Question: {input}
Thought:{agent_scratchpad}"""
