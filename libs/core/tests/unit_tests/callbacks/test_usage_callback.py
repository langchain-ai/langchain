from typing import Any

from langchain_core.callbacks import (
    UsageMetadataCallbackHandler,
    get_usage_metadata_callback,
)
from langchain_core.callbacks.usage import _usage_metadata_callback_var
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
    add_usage,
)
from langchain_core.outputs import ChatResult
from langchain_core.tracers.context import _configure_hooks, register_configure_hook

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


# ---------------------------------------------------------------------------
# Bug regression tests: _configure_hooks must not grow across repeated calls
# (https://github.com/langchain-ai/langchain/issues/32300)
# ---------------------------------------------------------------------------


def test_configure_hooks_no_accumulation_sequential() -> None:
    """_configure_hooks must not grow with repeated sequential calls."""
    # _usage_metadata_callback_var is registered once at module import time.
    # Capture the hook count after the first registration has already happened.
    hook_count_before = len(_configure_hooks)

    # Many sequential uses must not append new entries.
    for _ in range(10):
        with get_usage_metadata_callback():
            pass

    assert len(_configure_hooks) == hook_count_before, (
        f"_configure_hooks grew from {hook_count_before} to {len(_configure_hooks)} "
        "after repeated calls to get_usage_metadata_callback — hook accumulation bug."
    )


def test_configure_hooks_no_accumulation_nested() -> None:
    """_configure_hooks must not grow with nested calls."""
    hook_count_before = len(_configure_hooks)

    with (
        get_usage_metadata_callback(),
        get_usage_metadata_callback(),
        get_usage_metadata_callback(),
    ):
        pass

    assert len(_configure_hooks) == hook_count_before


def test_usage_metadata_callback_var_is_module_level() -> None:
    """The ContextVar used by get_usage_metadata_callback is the module-level singleton.

    Verifies that there is exactly ONE entry for _usage_metadata_callback_var in
    _configure_hooks, regardless of how many times get_usage_metadata_callback is used.
    """
    # Count entries that correspond to the module-level var.
    matching = [
        var for var, _, _, _ in _configure_hooks if var is _usage_metadata_callback_var
    ]
    assert len(matching) == 1, (
        f"Expected exactly 1 hook entry for _usage_metadata_callback_var, "
        f"got {len(matching)}."
    )


def test_configure_hooks_idempotent_registration() -> None:
    """register_configure_hook is idempotent: re-registering the same var is a no-op."""
    hook_count_before = len(_configure_hooks)
    # Attempt to register the already-registered module-level var again.
    register_configure_hook(_usage_metadata_callback_var, inheritable=True)
    assert len(_configure_hooks) == hook_count_before, (
        "register_configure_hook added a duplicate entry for an already-registered var."
    )


def test_nested_usage_callback_inner_does_not_pollute_outer() -> None:
    """Nested context managers track usage independently; outer is restored on exit."""
    llm = FakeChatModelWithResponseMetadata(
        messages=iter(messages), model_name="test_model"
    )

    with get_usage_metadata_callback() as outer_cb:
        _ = llm.invoke("Message 1")  # usage1 → outer_cb

        with get_usage_metadata_callback() as inner_cb:
            _ = llm.invoke("Message 2")  # usage2 → inner_cb only
            assert inner_cb.usage_metadata == {"test_model": usage2}
            # outer_cb must not have received the inner invocation
            assert outer_cb.usage_metadata == {"test_model": usage1}

        # After inner context exits, outer_cb resumes collecting.
        _ = llm.invoke("Message 3")  # usage3 → outer_cb
        assert outer_cb.usage_metadata == {"test_model": add_usage(usage1, usage3)}, (
            "Outer callback was not properly restored after inner context exited."
        )


def test_multiple_llm_instances_share_callback() -> None:
    """Multiple LLM instances all report to the same callback inside the context."""
    llm_1 = FakeChatModelWithResponseMetadata(
        messages=iter(messages[:2]), model_name="model_a"
    )
    llm_2 = FakeChatModelWithResponseMetadata(
        messages=iter(messages[2:4]), model_name="model_b"
    )

    with get_usage_metadata_callback() as cb:
        _ = llm_1.invoke("hello")  # usage1
        _ = llm_2.invoke("hello")  # usage3
        assert cb.usage_metadata == {
            "model_a": usage1,
            "model_b": usage3,
        }
