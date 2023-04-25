import tempfile
from typing import Any, List

from langchain.memory.chat_message_histories import (
    ChatMessageHistory,
    FileChatMessageHistory,
)
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain.schema import (
    BaseChatMessageHistory,
    BaseLanguageModel,
    BaseMessage,
    Generation,
    LLMResult,
)


# Define a custom class that inherits from BaseLanguageModel
class MockLanguageModel(BaseLanguageModel):
    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        return sum(len(message.content) for message in messages)

    def get_num_tokens(self, text: str) -> int:
        return len(text)

    def generate_prompt(self, prompts: Any, stop: Any = None) -> LLMResult:
        generations = [[Generation(text="mock summary")]]
        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts: Any, stop: Any = None) -> LLMResult:
        return self.generate_prompt(prompts, stop)


def helper_test_prune_messages(chat_memory: BaseChatMessageHistory) -> None:
    llm_mock = MockLanguageModel()
    summary_buffer = ConversationSummaryBufferMemory(
        chat_memory=chat_memory,
        llm=llm_mock,
        input_key="input",
        max_token_limit=200,
    )

    # Test the pruning mechanism
    summary_buffer.save_context(
        {
            "input": "Please, tell me a programmer joke",
        },
        {
            "response": (
                "Why do programmers prefer dark mode? " "Because light attracts bugs!"
            )
        },
    )
    summary_buffer.save_context(
        {
            "input": "plz tell me another joke",
        },
        {
            "response": (
                "Sure, here's another one: Why do programmers always "
                "mix up Halloween and Christmas? Because Oct 31 equals Dec 25!"
            )
        },
    )
    history = summary_buffer.load_memory_variables(inputs={"input": ""})["history"]
    print(history)
    history_length = summary_buffer.llm.get_num_tokens(text=history)
    assert history_length < summary_buffer.max_token_limit


def test_prune_messages_with_file_memory_history() -> None:
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        # Test ConversationSummaryBufferMemory's pruning with FileChatMessageHistory
        chat_memory = FileChatMessageHistory(file_path=temp_file.name)
        helper_test_prune_messages(chat_memory)


def test_prune_messages_with_in_memory_history() -> None:
    # Test ConversationSummaryBufferMemory's pruning with ChatMessageHistory
    chat_memory = ChatMessageHistory()
    helper_test_prune_messages(chat_memory)
