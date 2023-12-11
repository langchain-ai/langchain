from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .chat import chat
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

librarian_prompt_no_history = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """
            You are a librarian at cohere community library. Your job is to
            help recommend people books to read based on their interests and
            preferences. You also give information about the library.

            The library opens at 8am and closes at 9pm daily. It is closed on
            Sundays.

            If people speak too loud, you should tell them to shh

            Please answer the following message:
            """
        ),
        HumanMessagePromptTemplate.from_template("{message}"),
    ]
)

library_info = librarian_prompt_no_history | chat
