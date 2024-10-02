from langchain_community.callbacks import __all__, _module_lookup

EXPECTED_ALL = [
    "AimCallbackHandler",
    "ArgillaCallbackHandler",
    "ArizeCallbackHandler",
    "PanelCallbackHandler",
    "PromptLayerCallbackHandler",
    "ArthurCallbackHandler",
    "ClearMLCallbackHandler",
    "CometCallbackHandler",
    "ContextCallbackHandler",
    "HumanApprovalCallbackHandler",
    "InfinoCallbackHandler",
    "MlflowCallbackHandler",
    "LLMonitorCallbackHandler",
    "OpenAICallbackHandler",
    "LLMThoughtLabeler",
    "StreamlitCallbackHandler",
    "WandbCallbackHandler",
    "WhyLabsCallbackHandler",
    "get_openai_callback",
    "wandb_tracing_enabled",
    "FlyteCallbackHandler",
    "SageMakerCallbackHandler",
    "LabelStudioCallbackHandler",
    "TrubricsCallbackHandler",
    "FiddlerCallbackHandler",
    "UpTrainCallbackHandler",
    "UpstashRatelimitError",
    "UpstashRatelimitHandler",
]


def test_all_imports() -> None:
    """Test that __all__ is correctly set."""
    assert set(__all__) == set(EXPECTED_ALL)
    assert set(__all__) == set(_module_lookup.keys())
