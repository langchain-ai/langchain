from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser

from .prompts import answer_prompt
from .retriever_agent import executor

prompt = ChatPromptTemplate.from_template(answer_prompt)

model = ChatAnthropic(model="claude-2", temperature=0, max_tokens_to_sample=1000)

chain = (
    {"query": lambda x: x["query"], "information": executor | (lambda x: x["output"])}
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for the inputs to be used in the playground


class Inputs(BaseModel):
    query: str


chain = chain.with_types(input_type=Inputs)
