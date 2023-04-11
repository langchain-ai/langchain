import json
import pathlib
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    TypedDict,
    Union,
    cast,
)

from langchain.agents import BaseSingleActionAgent
from langchain.callbacks import get_callback_manager
from langchain.callbacks.tracers.base import SharedTracer
from langchain.callbacks.tracers.schemas import (
    ChainRun,
    LLMRun,
    ToolRun,
    TracerSession,
)

if TYPE_CHECKING:
    from wandb import Settings as WBSettings
    from wandb.integration.langchain.media_types import LangChainModelTrace
    from wandb.integration.langchain.schema import BaseRunSpan
    from wandb.wandb_run import Run as WBRun

    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.callbacks.tracers.schemas import (
        BaseRun,
        TracerSessionCreate,
    )
    from langchain.chains.base import Chain
    from langchain.llms.base import BaseLLM
    from langchain.schema import BaseLanguageModel
    from langchain.tools.base import BaseTool


def import_wandb() -> Any:
    try:
        import wandb  # noqa: F401
    except ImportError:
        raise ImportError(
            "To use the wandb callback manager you need to have the `wandb` python "
            "package installed. Please install it with `pip install wandb`"
        )
    return wandb


class WandbRunArgs(TypedDict):
    job_type: Optional[str]
    dir: Union[str, pathlib.Path, None]
    config: Union[Dict, str, None]
    project: Optional[str]
    entity: Optional[str]
    reinit: Optional[bool]
    tags: Optional[Sequence]
    group: Optional[str]
    name: Optional[str]
    notes: Optional[str]
    magic: Optional[Union[dict, str, bool]]
    config_exclude_keys: Optional[List[str]]
    config_include_keys: Optional[List[str]]
    anonymous: Optional[str]
    mode: Optional[str]
    allow_val_change: Optional[bool]
    resume: Optional[Union[bool, str]]
    force: Optional[bool]
    tensorboard: Optional[bool]
    sync_tensorboard: Optional[bool]
    monitor_gym: Optional[bool]
    save_code: Optional[bool]
    id: Optional[str]
    settings: Union["WBSettings", Dict[str, Any], None]


class WandbTracer(SharedTracer):
    """Callback Handler that logs to Weights and Biases.

    Parameters:
        run_args (dict): The arguments to pass to wandb.init().

    This handler will log the model architecture and run traces to Weights and Biases.
    """

    _run: Optional["WBRun"] = None
    _run_args: Optional[WandbRunArgs] = None

    @classmethod
    def watch_all(
        cls,
        run_args: Optional[WandbRunArgs] = None,
        additional_handlers: list["BaseCallbackHandler"] = [],
    ) -> "WandbTracer":
        """Sets up a WandbTracer and makes it the default handler. To use W&B to
        monitor all LangChain activity, simply call this function at the top of
        the notebook or script:
        ```
        from langchain.callbacks.wandb_tracer import WandbTracer
        WandbTracer.watch_all()
        # ...
        # end of notebook:
        WandbTracer.stop_watch()
        ```

        It is safe to call this over and over without any side effects. Users
        can pas new run_args which will trigger a new run to be created.

        Currently users would need to do something like:
        ```
        tracer = WandbTracer()
        manager = get_callback_manager()
        manager.set_handlers([tracer])
        ```

        which is a bit tedious. This is a one liner for users to globally
        monitor their LangChain activity.
        """
        tracer = cls()
        tracer.init(run_args)
        tracer.load_session("")
        manager = get_callback_manager()
        manager.set_handlers([tracer] + additional_handlers)
        return tracer

    @staticmethod
    def stop_watch() -> None:
        if WandbTracer._instance:
            cast(WandbTracer, WandbTracer._instance).finish()
            manager = get_callback_manager()
            manager.set_handlers([])

    def init(self, run_args: Optional[WandbRunArgs] = None) -> None:
        """Initialize wandb if it has not been initialized."""
        # Load in wandb symbols
        wandb = import_wandb()
        from wandb.integration.langchain.media_types import print_wandb_init_message

        # We only want to start a new run if the run args differ. This will
        # reduce the number of W&B runs created, which is more ideal in a
        # notebook setting
        if (
            wandb.run is not None
            and self._run is not None
            and json.dumps(self._run_args, sort_keys=True)
            == json.dumps(run_args, sort_keys=True)
        ):
            print_wandb_init_message(self._run.settings.run_url)
            return
        self._run_args = run_args
        self._run = None

        # Make a shallow copy of the run args so we don't modify the original
        run_args = run_args or {}  # type: ignore
        run_args: dict = {**run_args}  # type: ignore

        # Prefer to run in silent mode since W&B has a lot of output
        # which can be undesirable when dealing with text-based models.
        if "settings" not in run_args:  # type: ignore
            run_args["settings"] = {"silent": True}  # type: ignore

        # Start the run and add the stream table
        self._run = wandb.init(**run_args)
        print_wandb_init_message(self._run.settings.run_url)

    def finish(self) -> None:
        """Waits for W&B data to upload. It is recommended to call this function
        before terminating the kernel or python script."""
        if self._run is not None:
            url = self._run.settings.run_url
            self._run.finish()
            import_wandb()
            from wandb.integration.langchain.media_types import (
                print_wandb_finish_message,
            )

            print_wandb_finish_message(url)
        else:
            print("W&B run not started. Skipping.")

    def _log_trace(self, model_trace: "LangChainModelTrace") -> None:
        if self._run:
            self._run.log({"_langchain_model_trace": model_trace})

    ###  Start of required methods
    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""
        return None

    def _persist_run(self, run: "BaseRun") -> None:
        """Persist a run."""
        import_wandb()
        from wandb.integration.langchain.media_types import LangChainModelTrace

        self._log_trace(
            LangChainModelTrace(
                trace_span=_convert_lc_run_to_wb_span(run),
                model_dict=_safe_maybe_model_dict(_get_span_producing_object(run)),
            )
        )

    def _persist_session(
        self, session_create: "TracerSessionCreate"
    ) -> "TracerSession":
        """Persist a session."""
        return TracerSession(id=1, **session_create.dict())

    def load_session(self, session_name: str) -> "TracerSession":
        """Load a session from the tracer."""
        self._session = TracerSession(id=1)
        return self._session

    def load_default_session(self) -> "TracerSession":
        """Load the default tracing session and set it as the Tracer's session."""
        self._session = TracerSession(id=1)
        return self._session

    def _add_child_run(
        self,
        parent_run: Union["ChainRun", "ToolRun"],
        child_run: Union["LLMRun", "ChainRun", "ToolRun"],
    ) -> None:
        """Add child run to a chain run or tool run."""
        parent_run.child_runs.append(child_run)

    ## End of required methods


def _get_span_producing_object(run: "BaseRun") -> Any:
    return run.serialized.get("_self")


def _convert_lc_run_to_wb_span(run: "BaseRun") -> "BaseRunSpan":
    import_wandb()
    from wandb.integration.langchain.schema import (
        BaseRunSpan,
        ChainRunSpan,
        LLMResponse,
        LLMRunSpan,
        ToolRunSpan,
    )

    if isinstance(run, LLMRun):
        return LLMRunSpan(
            id=run.id,
            start_time=run.start_time,
            end_time=run.end_time,
            execution_order=run.execution_order,
            extra=run.extra,
            session_id=run.session_id,
            error=run.error,
            span_component_name=run.serialized.get("name"),
            prompt_responses=[
                LLMResponse(
                    prompt=prompt,
                    generation=run.response.generations[ndx][0].text
                    if run.response is not None
                    and len(run.response.generations) > ndx
                    and run.response.generations[ndx] is not None
                    and len(run.response.generations[ndx]) > 0
                    else None,
                )
                for ndx, prompt in enumerate(run.prompts)
            ],
            llm_output=run.response.llm_output if run.response is not None else None,
        )
    elif isinstance(run, ChainRun):
        return ChainRunSpan(
            id=run.id,
            start_time=run.start_time,
            end_time=run.end_time,
            execution_order=run.execution_order,
            session_id=run.session_id,
            extra=run.extra,
            error=run.error,
            span_component_name=run.serialized.get("name"),
            inputs=run.inputs,
            outputs=run.outputs,
            child_runs=[
                _convert_lc_run_to_wb_span(child_run) for child_run in run.child_runs
            ],
            is_agent=isinstance(_get_span_producing_object(run), BaseSingleActionAgent),
        )
    elif isinstance(run, ToolRun):
        return ToolRunSpan(
            id=run.id,
            start_time=run.start_time,
            end_time=run.end_time,
            execution_order=run.execution_order,
            extra=run.extra,
            session_id=run.session_id,
            error=run.error,
            span_component_name=run.serialized.get("name"),
            tool_input=run.tool_input,
            output=run.output,
            action=run.action,
            child_runs=[
                _convert_lc_run_to_wb_span(child_run) for child_run in run.child_runs
            ],
        )
    else:
        return BaseRunSpan(
            id=run.id,
            start_time=run.start_time,
            end_time=run.end_time,
            execution_order=run.execution_order,
            session_id=run.session_id,
            error=run.error,
            span_component_name=run.serialized.get("name"),
            span_type=None,
            extra=None,
        )


def _safe_maybe_model_dict(
    model: Union["BaseLanguageModel", "BaseLLM", "BaseTool", "Chain"]
) -> Optional[dict]:
    """Returns the model dict if possible, otherwise returns None.
    Given that Models are all user defined, this operation is not always possible.
    """
    data = None
    message = None
    try:
        data = model.dict()
    except Exception as e:
        message = str(e)
        pass


    if data is None and hasattr(model, "agent"):
        try:
            data = model.agent.dict()
        except Exception:
            message = str(e)

    if data is None and message is not None:
        print(f"Could not serialize model: {message}")

    return data
