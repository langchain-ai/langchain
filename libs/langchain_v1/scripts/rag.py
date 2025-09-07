from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from langchain.agents import create_agent
from langchain.agents.middleware.rag import RAGMiddleware
from langchain.chat_models.fake import FakeToolCallingModel
from langchain_core.retrievers import BaseRetriever

tool_calls = [[{"args": {}, "id": "1", "name": "handoff_to_foo2"}], []]

class FakeRetriever(BaseRetriever):

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[
        Document]:
        return [Document(page_content="foo")]


model = FakeToolCallingModel()
middleware = RAGMiddleware.from_retriever(FakeRetriever(), "foo")
agent = create_agent(model, [], middleware=[middleware])
print(agent.get_graph())
for s in agent.stream({"messages": [{"role": "user", "content": "hi"}]}, stream_mode="debug"):
    print(s)
