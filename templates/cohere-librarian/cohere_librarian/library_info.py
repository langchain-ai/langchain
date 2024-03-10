from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from .chat import chat

librarian_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """
            You are a librarian at cohere community library. Your job is to
            help recommend people books to read based on their interests and
            preferences. You also give information about the library.

            The library opens at 8am and closes at 9pm daily. It is closed on
            Sundays.

            Please answer the following message:
            """
        ),
        HumanMessagePromptTemplate.from_template("{message}"),
    ]
)

library_info = librarian_prompt | chat
