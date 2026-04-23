"""Test that RunnableBinding warns when pre-bound tools will be dropped.

See: https://github.com/langchain-ai/langchain/issues/35320
"""

import warnings

from langchain_core.runnables import RunnableBinding, RunnableLambda


def _make_base() -> RunnableLambda:
    base = RunnableLambda(lambda x: x)
    base.with_structured_output = lambda *args, **kwargs: "chain"  # type: ignore[attr-defined]
    return base


def test_warns_when_tools_pre_bound() -> None:
    base = _make_base()
    binding = RunnableBinding(
        bound=base, kwargs={"tools": [{"type": "web_search"}]}
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        binding.with_structured_output("Schema")
        assert len(w) == 1
        assert "pre-bound" in str(w[0].message)
        assert "tools" in str(w[0].message)


def test_warns_when_tool_choice_pre_bound() -> None:
    base = _make_base()
    binding = RunnableBinding(
        bound=base, kwargs={"tool_choice": "auto"}
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        binding.with_structured_output("Schema")
        assert len(w) == 1
        assert "tool_choice" in str(w[0].message)


def test_warns_when_both_pre_bound() -> None:
    base = _make_base()
    binding = RunnableBinding(
        bound=base,
        kwargs={"tools": [{"type": "web_search"}], "tool_choice": "auto"},
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        binding.with_structured_output("Schema")
        assert len(w) == 1
        assert "tools" in str(w[0].message)
        assert "tool_choice" in str(w[0].message)


def test_no_warning_when_no_conflicting_kwargs() -> None:
    base = _make_base()
    binding = RunnableBinding(bound=base, kwargs={"stop": ["x"]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        binding.with_structured_output("Schema")
        assert len(w) == 0


def test_forwards_to_bound_model() -> None:
    base = _make_base()
    binding = RunnableBinding(bound=base, kwargs={"stop": ["x"]})
    result = binding.with_structured_output("Schema")
    assert result == "chain"