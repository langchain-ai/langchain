"""**Callback handlers** allow listening to events in LangChain.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> <name>CallbackHandler  # Example: AimCallbackHandler
"""
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.callbacks.aim_callback import (
        AimCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.argilla_callback import (
        ArgillaCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.arize_callback import (
        ArizeCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.arthur_callback import (
        ArthurCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.clearml_callback import (
        ClearMLCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.comet_ml_callback import (
        CometCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.context_callback import (
        ContextCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.fiddler_callback import (
        FiddlerCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.flyte_callback import (
        FlyteCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.human import (
        HumanApprovalCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.infino_callback import (
        InfinoCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.labelstudio_callback import (
        LabelStudioCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.llmonitor_callback import (
        LLMonitorCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.manager import (  # noqa: F401
        get_openai_callback,
        wandb_tracing_enabled,
    )
    from langchain_community.callbacks.mlflow_callback import (
        MlflowCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.openai_info import (
        OpenAICallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.promptlayer_callback import (
        PromptLayerCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.sagemaker_callback import (
        SageMakerCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.streamlit import (  # noqa: F401
        LLMThoughtLabeler,
        StreamlitCallbackHandler,
    )
    from langchain_community.callbacks.trubrics_callback import (
        TrubricsCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.wandb_callback import (
        WandbCallbackHandler,  # noqa: F401
    )
    from langchain_community.callbacks.whylabs_callback import (
        WhyLabsCallbackHandler,  # noqa: F401
    )


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


__all__ = [
    "AimCallbackHandler",
    "ArgillaCallbackHandler",
    "ArizeCallbackHandler",
    "ArthurCallbackHandler",
    "ClearMLCallbackHandler",
    "CometCallbackHandler",
    "ContextCallbackHandler",
    "FiddlerCallbackHandler",
    "FlyteCallbackHandler",
    "HumanApprovalCallbackHandler",
    "InfinoCallbackHandler",
    "LLMThoughtLabeler",
    "LLMonitorCallbackHandler",
    "LabelStudioCallbackHandler",
    "MlflowCallbackHandler",
    "OpenAICallbackHandler",
    "PromptLayerCallbackHandler",
    "SageMakerCallbackHandler",
    "StreamlitCallbackHandler",
    "TrubricsCallbackHandler",
    "WandbCallbackHandler",
    "WhyLabsCallbackHandler",
    "get_openai_callback",
    "wandb_tracing_enabled",
]
