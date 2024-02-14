# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate

templ1 = """You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
Given a piece of text, you must come up with a question and answer pair that can be used to test a student's reading comprehension abilities.
When coming up with this question/answer pair, you must respond in the following format:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```

Everything between the ``` must be valid json.
"""
templ2 = """Please come up with a question/answer pair, in the specified JSON format, for the following text:
----------------
{text}"""
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(templ1),
        HumanMessagePromptTemplate.from_template(templ2),
    ]
)
templ = """You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
Given a piece of text, you must come up with a question and answer pair that can be used to test a student's reading comprehension abilities.
When coming up with this question/answer pair, you must respond in the following format:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```

Everything between the ``` must be valid json.

Please come up with a question/answer pair, in the specified JSON format, for the following text:
----------------
{text}"""
PROMPT = PromptTemplate.from_template(templ)

PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
