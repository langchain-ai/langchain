from typing import Any

from langchain_core.callbacks import (
    UsageMetadataCallbackHandler,
    get_usage_metadata_callback,
)
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
    add_usage,
)
from langchain_core.outputs import ChatResult

usage1 = UsageMetadata(
    input_tokens=1,
    output_tokens=2,
    total_tokens=3,
)
usage2 = UsageMetadata(
    input_tokens=4,
    output_tokens=5,
    total_tokens=9,
)
usage3 = UsageMetadata(
    input_tokens=10,
    output_tokens=20,
    total_tokens=30,
    input_token_details=InputTokenDetails(audio=5),
    output_token_details=OutputTokenDetails(reasoning=10),
)
usage4 = UsageMetadata(
    input_tokens=5,
    output_tokens=10,
    total_tokens=15,
    input_token_details=InputTokenDetails(audio=3),
    output_token_details=OutputTokenDetails(reasoning=5),
)
messages = [
    AIMessage("Response 1", usage_metadata=usage1),
    AIMessage("Response 2", usage_metadata=usage2),
    AIMessage("Response 3", usage_metadata=usage3),
    AIMessage("Response 4", usage_metadata=usage4),
]


class FakeChatModelWithResponseMetadata(GenericFakeChatModel):
    model_name: str

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        result = super()._generate(*args, **kwargs)
        result.generations[0].message.response_metadata = {
            "model_name": self.model_name
        }
        return result


def test_usage_callback() -> None:
    llm = FakeChatModelWithResponseMetadata(
        messages=iter(messages), model_name="test_model"
    )

    # Test context manager
    with get_usage_metadata_callback() as cb:
        _ = llm.invoke("Message 1")
        _ = llm.invoke("Message 2")
        total_1_2 = add_usage(usage1, usage2)
        assert cb.usage_metadata == {"test_model": total_1_2}
        _ = llm.invoke("Message 3")
        _ = llm.invoke("Message 4")
        total_3_4 = add_usage(usage3, usage4)
        assert cb.usage_metadata == {"test_model": add_usage(total_1_2, total_3_4)}

    # Test via config
    llm = FakeChatModelWithResponseMetadata(
        messages=iter(messages[:2]), model_name="test_model"
    )
    callback = UsageMetadataCallbackHandler()
    _ = llm.batch(["Message 1", "Message 2"], config={"callbacks": [callback]})
    assert callback.usage_metadata == {"test_model": total_1_2}

    # Test multiple models
    llm_1 = FakeChatModelWithResponseMetadata(
        messages=iter(messages[:2]), model_name="test_model_1"
    )
    llm_2 = FakeChatModelWithResponseMetadata(
        messages=iter(messages[2:4]), model_name="test_model_2"
    )
    callback = UsageMetadataCallbackHandler()
    _ = llm_1.batch(["Message 1", "Message 2"], config={"callbacks": [callback]})
    _ = llm_2.batch(["Message 3", "Message 4"], config={"callbacks": [callback]})
    assert callback.usage_metadata == {
        "test_model_1": total_1_2,
        "test_model_2": total_3_4,
    }


async def test_usage_callback_async() -> None:
    llm = FakeChatModelWithResponseMetadata(
        messages=iter(messages), model_name="test_model"
    )

    # Test context manager
    with get_usage_metadata_callback() as cb:
        _ = await llm.ainvoke("Message 1")
        _ = await llm.ainvoke("Message 2")
        total_1_2 = add_usage(usage1, usage2)
        assert cb.usage_metadata == {"test_model": total_1_2}
        _ = await llm.ainvoke("Message 3")
        _ = await llm.ainvoke("Message 4")
        total_3_4 = add_usage(usage3, usage4)
        assert cb.usage_metadata == {"test_model": add_usage(total_1_2, total_3_4)}

    # Test via config
    llm = FakeChatModelWithResponseMetadata(
        messages=iter(messages[:2]), model_name="test_model"
    )
    callback = UsageMetadataCallbackHandler()
    _ = await llm.abatch(["Message 1", "Message 2"], config={"callbacks": [callback]})
    assert callback.usage_metadata == {"test_model": total_1_2}
