from typing import Optional

from langchain.chains import create_structured_output_runnable
from langchain_community.chat_models.fake import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field


def test_structured_output_tools():
    class Dog(BaseModel):
        '''Identifying information about a dog.'''

        name: str = Field(..., description="The dog's name")
        color: str = Field(..., description="The dog's color")
        fav_food: Optional[str] = Field(None, description="The dog's favorite food")

    responses = [
        AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_qtDBDM46DpUsPA3BC0hucEUi', 'function': {'arguments': '{"output": {"name": "Harry", "color": "brown", "fav_food": "chicken"}}', 'name': '_OutputFormatter'}, 'type': 'function'}, {'id': 'call_kIPim7iMCL1Ekk61jCenP5Kd', 'function': {'arguments': '{"output": {"name": "Joe", "color": "black", "fav_food": "dog food"}}', 'name': '_OutputFormatter'}, 'type': 'function'}]})
    ]
    llm = FakeMessagesListChatModel(responses=responses)
    structured_llm = create_structured_output_runnable(Dog, llm, mode="openai-tools")
    result = structured_llm.invoke(
        "Harry was a chubby brown beagle who loved chicken. "
        "Joe was a black lab who loved dog food."
    )
    assert [Dog(name='Harry', color='brown', fav_food='chicken'), Dog(name='Joe', color='black', fav_food='dog food')] == result
