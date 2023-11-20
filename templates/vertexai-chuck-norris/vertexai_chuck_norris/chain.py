from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate

_prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about Chuck Norris and {text}"
)
_model = ChatVertexAI()

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model
