PD_PREFIX = """
Ты работаешь с объектом dataframe Xorbits в Python.
Перед импортом Numpy или Pandas в текущий скрипт,
не забудь импортировать версию библиотеки xorbits вместо оригинальной.
Чтобы импортировать версию xorbits Numpy, замени оригинальное выражение импорта
`import pandas as pd` на `import xorbits.pandas as pd`. 
Название входных данных - `data`.
Ты должен использовать инструменты ниже, чтобы ответить на заданный тебе вопрос:"""

PD_SUFFIX = """
Это результат `print(data)`:
{data}

Начни!
Question: {input}
{agent_scratchpad}"""

NP_PREFIX = """
Ты работаешь с объектом ndarray Xorbits в Python.
Перед импортом Numpy в текущий скрипт,
не забудь импортировать версию библиотеки xorbits вместо оригинальной.
Чтобы импортировать версию xorbits Numpy, замени оригинальное выражение импорта
`import numpy as np` на `import xorbits.numpy as np`.
Название входных данных - `data`.
Ты должен использовать инструменты ниже, чтобы ответить на заданный тебе вопрос:"""

NP_SUFFIX = """
Это результат `print(data)`:
{data}

Начни!
Question: {input}
{agent_scratchpad}"""
