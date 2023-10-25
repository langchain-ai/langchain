from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

# Function output schema
class Overview(BaseModel):
    """Summary, language, and keywords for input text"""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")

# Function definition
model = ChatOpenAI()
function = [convert_pydantic_to_openai_function(Overview)]
chain = model.bind(
    functions=function, function_call={"name": "Overview"}
).with_types(input_type=str)
