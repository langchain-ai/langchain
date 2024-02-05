from langchain_community.chat_models import ChatAnthropic, ChatCohere, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate user input into pirate speak",
        ),
        ("human", "{text}"),
    ]
)
_model = ChatOpenAI().configurable_alternatives(
    ConfigurableField(id="llm_provider"),
    default_key="openai",
    anthropic=ChatAnthropic,
    cohere=ChatCohere,
)

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model
