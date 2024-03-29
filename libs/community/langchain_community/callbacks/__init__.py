"""**Callback handlers** allow listening to events in LangChain.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> <name>CallbackHandler  # Example: AimCallbackHandler
"""
import importlib
from typing import Any

_module_lookup = {
    "AimCallbackHandler": "langchain_community.callbacks.aim_callback",
    "ArgillaCallbackHandler": "langchain_community.callbacks.argilla_callback",
    "ArizeCallbackHandler": "langchain_community.callbacks.arize_callback",
    "ArthurCallbackHandler": "langchain_community.callbacks.arthur_callback",
    "ClearMLCallbackHandler": "langchain_community.callbacks.clearml_callback",
    "CometCallbackHandler": "langchain_community.callbacks.comet_ml_callback",
    "ContextCallbackHandler": "langchain_community.callbacks.context_callback",
    "FiddlerCallbackHandler": "langchain_community.callbacks.fiddler_callback",
    "FlyteCallbackHandler": "langchain_community.callbacks.flyte_callback",
    "HumanApprovalCallbackHandler": "langchain_community.callbacks.human",
    "InfinoCallbackHandler": "langchain_community.callbacks.infino_callback",
    "LLMThoughtLabeler": "langchain_community.callbacks.streamlit",
    "LLMonitorCallbackHandler": "langchain_community.callbacks.llmonitor_callback",
    "LabelStudioCallbackHandler": "langchain_community.callbacks.labelstudio_callback",
    "MlflowCallbackHandler": "langchain_community.callbacks.mlflow_callback",
    "OpenAICallbackHandler": "langchain_community.callbacks.openai_info",
    "PromptLayerCallbackHandler": "langchain_community.callbacks.promptlayer_callback",
    "SageMakerCallbackHandler": "langchain_community.callbacks.sagemaker_callback",
    "StreamlitCallbackHandler": "langchain_community.callbacks.streamlit",
    "TrubricsCallbackHandler": "langchain_community.callbacks.trubrics_callback",
    "WandbCallbackHandler": "langchain_community.callbacks.wandb_callback",
    "WhyLabsCallbackHandler": "langchain_community.callbacks.whylabs_callback",
    "get_openai_callback": "langchain_community.callbacks.manager",
    "wandb_tracing_enabled": "langchain_community.callbacks.manager",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
