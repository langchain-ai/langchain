from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


class Question(BaseModel):
    __root__: str


wiki = WikipediaAPIWrapper(top_k_results=5)
prompt = hub.pull("bagatur/chain-of-note-wiki")

llm = ChatAnthropic(model="claude-3-sonnet-20240229")


def format_docs(docs):
    return "\n\n".join(
        f"Wikipedia {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)
    )


chain = (
    {
        "passages": RunnableLambda(wiki.load) | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
).with_types(input_type=Question)
