"""Test @chain decorator type compatibility."""

from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig, chain


class TestChainDecorator:
    """Test @chain decorator with RunnableConfig parameter."""

    def test_chain_with_config_parameter(self) -> None:
        """Test that @chain works with functions accepting RunnableConfig."""

        @chain
        def to_plain_text(data: ChatPromptValue, config: RunnableConfig) -> str:
            if config["configurable"].get("upper_case", False):
                return data.to_string().upper()
            return data.to_string()

        prompt = ChatPromptValue(messages=[AIMessage("test message")])

        # Test upper case
        result = to_plain_text.invoke(
            prompt, config=RunnableConfig(configurable={"upper_case": True})
        )
        assert result == "AI: TEST MESSAGE"

        # Test normal case
        result = to_plain_text.invoke(
            prompt, config=RunnableConfig(configurable={"upper_case": False})
        )
        assert result == "AI: test message"

    def test_chain_backward_compatibility(self) -> None:
        """Test existing @chain usage still works."""

        @chain
        def simple_chain(data: str) -> str:
            return f"simple: {data}"

        result = simple_chain.invoke("test")
        assert result == "simple: test"
