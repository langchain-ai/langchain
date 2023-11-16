from langchain import hub
from langchain.chat_models import ChatAnthropic
from langchain.utilities import WikipediaAPIWrapper
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

wiki = WikipediaAPIWrapper(top_k_results=5)
prompt = hub.pull("bagatur/chain-of-note-wiki")

llm = ChatAnthropic(model="claude-2")

def format_docs(docs):
    return "\n\n".join(f"Wikipedia {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

chain = (
    {
        "passages": RunnableLambda(wiki.load) | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt | llm | StrOutputParser()
)