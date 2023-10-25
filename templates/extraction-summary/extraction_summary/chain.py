from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.utils.openai_functions import convert_pydantic_to_openai_function


# Function output schema
class Overview(BaseModel):
    """Summary, langugae, and keywords for input text"""

    summary: str = Field(description="Provide a concise summary of the content.")
    langugae: str = Field(
        description="Provide the languge that the content is written in."
    )
    keywords: str = Field(description="Provide keywords related to the content.")


# Function definition
model = ChatOpenAI()
overview_extraction_function = [convert_pydantic_to_openai_function(Overview)]
chain = model.bind(
    functions=overview_extraction_function, function_call={"name": "Overview"}
)
