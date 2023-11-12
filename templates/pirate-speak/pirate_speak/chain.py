from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate user input into pirate speak",
        ),
        ("human", "{text}"),
    ]
)
_model = ChatOpenAI()

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model
