from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain.schema import AIMessage, HumanMessage
from tests.unit_tests.llms.fake_llm import FakeLLM


class TestConversationSummaryBufferMemory:
    def test_no_buffer_yet(self) -> None:
        """Test ConversationSummaryBufferMemory when no inputs put in buffer yet."""
        memory = ConversationSummaryBufferMemory(llm=FakeLLM(), memory_key="baz")
        output = memory.load_memory_variables({})
        assert output == {"baz": ""}

    def test_buffer_only(self) -> None:
        """Test ConversationSummaryBufferMemory when only buffer."""
        memory = ConversationSummaryBufferMemory(llm=FakeLLM(), memory_key="baz")
        memory.save_context({"input": "bar"}, {"output": "foo"})
        assert memory.buffer == [
            HumanMessage(content="bar", additional_kwargs={}),
            AIMessage(content="foo", additional_kwargs={}),
        ]
        output = memory.load_memory_variables({})
        assert output == {"baz": "Human: bar\nAI: foo"}

    def test_summary(self) -> None:
        # TODO: There seems to be a discrepancy between the expected and actual
        #  values in the test_summary method. The memory.buffer assertion seems to be
        #  checking for AIMessage, HumanMessage, and AIMessage, whereas the
        #  memory.load_memory_variables assertion seems to be checking for a
        #  SystemMessage. It is unclear why these discrepancies exist in the test,
        #  and it may require further investigation into the implementation of the
        #  ConversationSummaryBufferMemory class and the FakeLLM class to determine
        #  the root cause of the issue.

        """Test ConversationSummaryBufferMemory when only buffer."""
        memory = ConversationSummaryBufferMemory(
            llm=FakeLLM(), memory_key="baz", max_token_limit=13
        )
        memory.save_context({"input": "bar"}, {"output": "foo"})
        memory.save_context({"input": "bar1"}, {"output": "foo1"})

        # TODO: unclear in test why we got AIMessage, HumanMessage, AIMessage
        # TODO: instead of HumanMessage, AIMessage, HumanMessage, AIMessage
        assert memory.buffer == [
            AIMessage(content="foo", additional_kwargs={}),
            HumanMessage(content="bar1", additional_kwargs={}),
            AIMessage(content="foo1", additional_kwargs={}),
        ]

        output = memory.load_memory_variables({})
        # TODO: unclear in test why we got system message
        assert output == {"baz": "System: foo\nAI: foo\nHuman: bar1\nAI: foo1"}
