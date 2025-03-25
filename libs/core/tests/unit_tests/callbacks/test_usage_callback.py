from itertools import cycle

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


def test_usage_callback() -> None:
    llm = GenericFakeChatModel(messages=cycle(messages))

    # Test context manager
    with get_usage_metadata_callback() as cb:
        _ = llm.invoke("Message 1")
        _ = llm.invoke("Message 2")
        total_1_2 = add_usage(usage1, usage2)
        assert cb.usage_metadata == total_1_2
        _ = llm.invoke("Message 3")
        _ = llm.invoke("Message 4")
        total_3_4 = add_usage(usage3, usage4)
        assert cb.usage_metadata == add_usage(total_1_2, total_3_4)

    # Test via config
    callback = UsageMetadataCallbackHandler()
    _ = llm.batch(["Message 1", "Message 2"], config={"callbacks": [callback]})
    assert callback.usage_metadata == total_1_2


async def test_usage_callback_async() -> None:
    llm = GenericFakeChatModel(messages=cycle(messages))

    # Test context manager
    with get_usage_metadata_callback() as cb:
        _ = await llm.ainvoke("Message 1")
        _ = await llm.ainvoke("Message 2")
        total_1_2 = add_usage(usage1, usage2)
        assert cb.usage_metadata == total_1_2
        _ = await llm.ainvoke("Message 3")
        _ = await llm.ainvoke("Message 4")
        total_3_4 = add_usage(usage3, usage4)
        assert cb.usage_metadata == add_usage(total_1_2, total_3_4)

    # Test via config
    callback = UsageMetadataCallbackHandler()
    _ = await llm.abatch(["Message 1", "Message 2"], config={"callbacks": [callback]})
    assert callback.usage_metadata == total_1_2
