from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import DuckDuckGoSearchAPIWrapper

template = """Answer the users question based only on the following context:

<context>
{context}
</context>

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0)

search = DuckDuckGoSearchAPIWrapper()


def retriever(query):
    return search.run(query)


template = """Provide a better search query for \
web search engine to answer the given question, end \
the queries with ’**’. Question: \
{x} Answer:"""
rewrite_prompt = ChatPromptTemplate.from_template(template)

# Parser to remove the `**`


def _parse(text):
    return text.strip("**")


rewriter = rewrite_prompt | ChatOpenAI(temperature=0) | StrOutputParser() | _parse

chain = (
    {
        "context": {"x": RunnablePassthrough()} | rewriter | retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Add input type for playground


class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
