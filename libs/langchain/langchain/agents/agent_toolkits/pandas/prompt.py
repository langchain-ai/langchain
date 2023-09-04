# flake8: noqa

PREFIX = """
Ты работаешь с pandas dataframe в Python. Имя dataframe - `df`.
Тебе следует использовать инструменты ниже, чтобы ответить на заданный вопрос:"""

MULTI_DF_PREFIX = """
Ты работаешь с {num_dfs} pandas dataframes в Python, которые называются df1, df2 и т.д. Тебе 
следует использовать инструменты ниже, чтобы ответить на заданный вопрос:"""

SUFFIX_NO_DF = """
Начни!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_DF = """
Это результат `print(df.head())`:
{df_head}

Начни!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_MULTI_DF = """
Это результат `print(df.head())` для каждого dataframe:
{dfs_head}

Начни!
Question: {input}
{agent_scratchpad}"""

PREFIX_FUNCTIONS = """
Ты работаешь с pandas dataframe в Python. Имя dataframe - `df`."""

MULTI_DF_PREFIX_FUNCTIONS = """
Ты работаешь с {num_dfs} pandas dataframes в Python, которые называются df1, df2 и т.д."""

FUNCTIONS_WITH_DF = """
Это результат `print(df.head())`:
{df_head}"""

FUNCTIONS_WITH_MULTI_DF = """
Это результат `print(df.head())` для каждого dataframe:
{dfs_head}"""
