from langchain_classic.schema.runnable.configurable import __all__

EXPECTED_ALL = [
    "DynamicRunnable",
    "RunnableConfigurableAlternatives",
    "RunnableConfigurableFields",
    "StrEnum",
    "make_options_spec",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
