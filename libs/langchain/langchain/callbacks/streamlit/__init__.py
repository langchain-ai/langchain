from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain_core.callbacks.base import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_community.callbacks import LLMThoughtLabeler
    from streamlit.delta_generator import DeltaGenerator


def StreamlitCallbackHandler(
    parent_container: DeltaGenerator,
    *,
    max_thought_containers: int = 4,
    expand_new_thoughts: bool = True,
    collapse_completed_thoughts: bool = True,
    thought_labeler: Optional[LLMThoughtLabeler] = None,
) -> BaseCallbackHandler:
    """Callback Handler that writes to a Streamlit app.

    This CallbackHandler is geared towards
    use with a LangChain Agent; it displays the Agent's LLM and tool-usage "thoughts"
    inside a series of Streamlit expanders.

    Parameters
    ----------
    parent_container
        The `st.container` that will contain all the Streamlit elements that the
        Handler creates.
    max_thought_containers
        The max number of completed LLM thought containers to show at once. When this
        threshold is reached, a new thought will cause the oldest thoughts to be
        collapsed into a "History" expander. Defaults to 4.
    expand_new_thoughts
        Each LLM "thought" gets its own `st.expander`. This param controls whether that
        expander is expanded by default. Defaults to True.
    collapse_completed_thoughts
        If True, LLM thought expanders will be collapsed when completed.
        Defaults to True.
    thought_labeler
        An optional custom LLMThoughtLabeler instance. If unspecified, the handler
        will use the default thought labeling logic. Defaults to None.

    Returns
    -------
    A new StreamlitCallbackHandler instance.

    Note that this is an "auto-updating" API: if the installed version of Streamlit
    has a more recent StreamlitCallbackHandler implementation, an instance of that class
    will be used.

    """
    # If we're using a version of Streamlit that implements StreamlitCallbackHandler,
    # delegate to it instead of using our built-in handler. The official handler is
    # guaranteed to support the same set of kwargs.
    try:
        from streamlit.external.langchain import StreamlitCallbackHandler

        # This is the official handler, so we can just return it.
        return StreamlitCallbackHandler(
            parent_container,
            max_thought_containers=max_thought_containers,
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
            thought_labeler=thought_labeler,
        )
    except ImportError:
        try:
            from langchain_community.callbacks.streamlit.streamlit_callback_handler import (  # noqa: E501
                StreamlitCallbackHandler as _InternalStreamlitCallbackHandler,
            )
        except ImportError as e:
            msg = (
                "To use the StreamlitCallbackHandler, please install "
                "langchain-community with `pip install langchain-community`."
            )
            raise ImportError(msg) from e

        return _InternalStreamlitCallbackHandler(
            parent_container,
            max_thought_containers=max_thought_containers,
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
            thought_labeler=thought_labeler,
        )
