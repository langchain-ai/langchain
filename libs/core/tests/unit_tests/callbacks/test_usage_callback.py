import warnings
from typing import Any

import pytest

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
from langchain_core.tracers.context import _configure_hooks

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


def test_configure_hooks_no_accumulation() -> None:
    """Regression test for https://github.com/langchain-ai/langchain/issues/32300.

    Each call to get_usage_metadata_callback() must NOT append a new entry to
    _configure_hooks. The hook is registered once at module load time via the
    module-level _usage_metadata_callback_var.
    """
    hooks_before = len(_configure_hooks)

    for _ in range(10):
        with get_usage_metadata_callback():
            pass

    hooks_after = len(_configure_hooks)
    assert hooks_after == hooks_before, (
        f"_configure_hooks grew from {hooks_before} to {hooks_after} entries after "
        "repeated calls to get_usage_metadata_callback(). Each call must not register "
        "a new hook."
    )

    # Sanity-check: the module-level var is registered exactly once.
    count = sum(
        1 for var, *_ in _configure_hooks if var is _usage_metadata_callback_var
    )
    assert count == 1


def test_name_parameter_deprecated() -> None:
    """Passing a custom `name` must emit a DeprecationWarning."""
    with (
        pytest.warns(DeprecationWarning, match="`name` parameter"),
        get_usage_metadata_callback(name="my_custom_name"),
    ):
        pass


def test_name_parameter_default_no_warning() -> None:
    """Calling with the default `name` must NOT emit any deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with get_usage_metadata_callback():
            pass


def test_nested_context_managers() -> None:
    """Inner context manager must not clobber the outer one on exit."""
    with get_usage_metadata_callback() as outer_cb:
        with get_usage_metadata_callback() as inner_cb:
            assert outer_cb is not inner_cb

        # After the inner CM exits, the outer handler must still be active.
        assert _usage_metadata_callback_var.get() is outer_cb

    # After the outer CM exits, the var must be back to its previous value (None).
    assert _usage_metadata_callback_var.get() is None
