import pytest

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.callbacks import usage as usage_module
from langchain_core.tracers import context as tracers_context


@pytest.fixture(autouse=True)
def restore_usage_registration_state() -> None:
    original_hooks = list(tracers_context._configure_hooks)
    original_vars = dict(usage_module._usage_metadata_callback_vars)
    try:
        yield
    finally:
        tracers_context._configure_hooks[:] = original_hooks
        usage_module._usage_metadata_callback_vars.clear()
        usage_module._usage_metadata_callback_vars.update(original_vars)


def hook_var_names() -> list[str]:
    return [var.name for var, _, _, _ in tracers_context._configure_hooks]


def test_repeated_default_usage_callback_registers_only_one_hook() -> None:
    base_hook_count = len(tracers_context._configure_hooks)

    for _ in range(5):
        with get_usage_metadata_callback() as cb:
            assert cb.usage_metadata == {}

    assert len(tracers_context._configure_hooks) == base_hook_count + 1
    assert usage_module._usage_metadata_callback_vars.keys() == {"usage_metadata_callback"}
    assert hook_var_names().count("usage_metadata_callback") == 1


def test_repeated_custom_usage_callback_name_registers_only_one_hook() -> None:
    base_hook_count = len(tracers_context._configure_hooks)

    for _ in range(4):
        with get_usage_metadata_callback("custom-usage") as cb:
            assert cb.usage_metadata == {}

    assert len(tracers_context._configure_hooks) == base_hook_count + 1
    assert usage_module._usage_metadata_callback_vars.keys() == {"custom-usage"}
    assert hook_var_names().count("custom-usage") == 1


def test_distinct_usage_callback_names_register_once_each() -> None:
    base_hook_count = len(tracers_context._configure_hooks)

    for name in ["alpha", "beta", "alpha", "gamma", "beta"]:
        with get_usage_metadata_callback(name):
            pass

    assert len(tracers_context._configure_hooks) == base_hook_count + 3
    assert usage_module._usage_metadata_callback_vars.keys() == {
        "alpha",
        "beta",
        "gamma",
    }
    assert hook_var_names().count("alpha") == 1
    assert hook_var_names().count("beta") == 1
    assert hook_var_names().count("gamma") == 1


def test_nested_same_name_restores_outer_callback_after_inner_exit() -> None:
    with get_usage_metadata_callback("shared") as outer:
        shared_var = usage_module._get_usage_metadata_callback_var("shared")
        assert shared_var.get() is outer

        with get_usage_metadata_callback("shared") as inner:
            assert shared_var.get() is inner
            assert inner is not outer

        assert shared_var.get() is outer

    assert shared_var.get() is None
    assert hook_var_names().count("shared") == 1


def test_nested_distinct_names_keep_callback_contexts_isolated() -> None:
    with get_usage_metadata_callback("outer") as outer:
        outer_var = usage_module._get_usage_metadata_callback_var("outer")
        inner_var = usage_module._get_usage_metadata_callback_var("inner")
        assert outer_var.get() is outer
        assert inner_var.get() is None

        with get_usage_metadata_callback("inner") as inner:
            assert outer_var.get() is outer
            assert inner_var.get() is inner

        assert outer_var.get() is outer
        assert inner_var.get() is None

    assert outer_var.get() is None
    assert inner_var.get() is None


def test_same_name_reuses_registered_context_var_but_not_handler_instance() -> None:
    with get_usage_metadata_callback("reused") as first:
        callback_var = usage_module._get_usage_metadata_callback_var("reused")
        assert callback_var.get() is first

    with get_usage_metadata_callback("reused") as second:
        assert callback_var.get() is second
        assert second is not first

    assert callback_var.get() is None
    assert hook_var_names().count("reused") == 1


def test_callback_var_is_reset_when_context_exits_with_exception() -> None:
    callback_var = usage_module._get_usage_metadata_callback_var("exception-reset")

    with pytest.raises(RuntimeError, match="boom"):
        with get_usage_metadata_callback("exception-reset") as cb:
            assert callback_var.get() is cb
            raise RuntimeError("boom")

    assert callback_var.get() is None


def test_callback_manager_hook_entries_match_registered_usage_names() -> None:
    names = ["usage-a", "usage-b", "usage-a", "usage-c"]
    for name in names:
        with get_usage_metadata_callback(name):
            pass

    registered_names = set(usage_module._usage_metadata_callback_vars)
    hook_names = {
        var.name
        for var, inheritable, handler_class, env_var in tracers_context._configure_hooks
        if inheritable and handler_class is None and env_var is None and var.name.startswith("usage-")
    }

    assert registered_names == {"usage-a", "usage-b", "usage-c"}
    assert hook_names == registered_names



def test_context_var_object_is_stable_per_name() -> None:
    alpha_var_first = usage_module._get_usage_metadata_callback_var("stable-alpha")
    alpha_var_second = usage_module._get_usage_metadata_callback_var("stable-alpha")
    beta_var = usage_module._get_usage_metadata_callback_var("stable-beta")

    assert alpha_var_first is alpha_var_second
    assert alpha_var_first is not beta_var


@pytest.mark.parametrize(
    ("name", "repeat_count"),
    [
        ("matrix-a", 1),
        ("matrix-a", 3),
        ("matrix-b", 2),
        ("matrix-c", 4),
    ],
)
def test_registration_growth_depends_on_unique_names_not_repetition(
    name: str, repeat_count: int
) -> None:
    base_hook_count = len(tracers_context._configure_hooks)
    seen_names: set[str] = set()

    for current_name, current_repeat_count in [
        ("matrix-a", 1),
        ("matrix-a", 3),
        ("matrix-b", 2),
        ("matrix-c", 4),
    ]:
        seen_names.add(current_name)
        for _ in range(current_repeat_count):
            with get_usage_metadata_callback(current_name):
                pass

    assert len(tracers_context._configure_hooks) == base_hook_count + len(seen_names)
    assert hook_var_names().count(name) == 1


def test_nested_same_name_can_reenter_multiple_times() -> None:
    callback_var = usage_module._get_usage_metadata_callback_var("triple-nested")

    with get_usage_metadata_callback("triple-nested") as outer:
        assert callback_var.get() is outer

        with get_usage_metadata_callback("triple-nested") as middle:
            assert callback_var.get() is middle

            with get_usage_metadata_callback("triple-nested") as inner:
                assert callback_var.get() is inner

            assert callback_var.get() is middle

        assert callback_var.get() is outer

    assert callback_var.get() is None


def test_distinct_named_callbacks_do_not_share_usage_state() -> None:
    with get_usage_metadata_callback("usage-left") as left:
        left.usage_metadata["left-model"] = {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
        }
        with get_usage_metadata_callback("usage-right") as right:
            right.usage_metadata["right-model"] = {
                "input_tokens": 4,
                "output_tokens": 5,
                "total_tokens": 9,
            }
            assert "left-model" not in right.usage_metadata

        assert "right-model" not in left.usage_metadata
        assert left.usage_metadata["left-model"]["total_tokens"] == 3
