"""A Tracer Implementation that records activity to Weights & Biases."""
try:
    import wandb  # noqa: F401
except ImportError:
    raise ImportError(
        "To use the WandbTracer you need to have the `wandb` python "
        "package installed. Please install it with `pip install wandb`"
    )


from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypedDict, Union

import wandb.util
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import ChainRun, LLMRun, ToolRun, TracerSession
from wandb.sdk.data_types import trace_tree
from wandb.sdk.lib import telemetry as wb_telemetry
from wandb.sdk.lib.paths import StrPath

if TYPE_CHECKING:
    from langchain.base_language import BaseLanguageModel
    from langchain.callbacks.tracers.schemas import BaseRun, TracerSessionBase
    from langchain.chains.base import Chain
    from langchain.llms.base import BaseLLM
    from langchain.tools.base import BaseTool
    from wandb import Settings as WBSettings
    from wandb.wandb_run import Run as WBRun

PRINT_WARNINGS = True


def print_wandb_init_message(run_url: str) -> None:
    wandb.termlog(
        f"Streaming LangChain activity to W&B at {run_url}\n"
        "`WandbTracer` is currently in beta.\n"
        "Please report any issues to https://github.com/wandb/wandb/issues with the tag `langchain`."
    )


def safely_convert_lc_run_to_wb_span(run: "BaseRun") -> Optional["trace_tree.Span"]:
    try:
        return _convert_lc_run_to_wb_span(run)
    except Exception as e:
        if PRINT_WARNINGS:
            wandb.termwarn(
                f"Skipping trace saving - unable to safely convert LangChain Run into W&B Trace due to: {e}"
            )
    return None


def safely_get_span_producing_model(run: "BaseRun") -> Any:
    try:
        return run.serialized.get("_self")
    except Exception as e:
        if PRINT_WARNINGS:
            wandb.termwarn(
                f"Skipping model saving - unable to safely retrieve LangChain model due to: {e}"
            )
    return None


def safely_convert_model_to_dict(
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
        if hasattr(model, "agent"):
            try:
                data = model.agent.dict()
            except Exception as e:
                message = str(e)

    if data is not None and not isinstance(data, dict):
        message = (
            f"Model's dict transformation resulted in {type(data)}, expected a dict."
        )
        data = None

    if data is not None:
        data = _replace_type_with_kind(data)
    else:
        if PRINT_WARNINGS:
            wandb.termwarn(
                f"Skipping model saving - unable to safely convert LangChain Model to dictionary due to: {message}"
            )

    return data


def _convert_lc_run_to_wb_span(run: "BaseRun") -> "trace_tree.Span":
    if isinstance(run, LLMRun):
        return _convert_llm_run_to_wb_span(run)
    elif isinstance(run, ChainRun):
        return _convert_chain_run_to_wb_span(run)
    elif isinstance(run, ToolRun):
        return _convert_tool_run_to_wb_span(run)
    else:
        return _convert_run_to_wb_span(run)


def _convert_llm_run_to_wb_span(run: "LLMRun") -> "trace_tree.Span":
    base_span = _convert_run_to_wb_span(run)

    if run.response is not None:
        base_span.add_attribute("llm_output", run.response.llm_output)
    base_span.results = [
        trace_tree.Result(
            inputs={"prompt": prompt},
            outputs={
                f"gen_{g_i}": gen.text
                for g_i, gen in enumerate(run.response.generations[ndx])
            }
            if (
                run.response is not None
                and len(run.response.generations) > ndx
                and len(run.response.generations[ndx]) > 0
            )
            else None,
        )
        for ndx, prompt in enumerate(run.prompts or [])
    ]
    base_span.span_kind = trace_tree.SpanKind.LLM

    return base_span


def _convert_chain_run_to_wb_span(run: "ChainRun") -> "trace_tree.Span":
    base_span = _convert_run_to_wb_span(run)

    base_span.results = [trace_tree.Result(inputs=run.inputs, outputs=run.outputs)]
    base_span.child_spans = [
        _convert_lc_run_to_wb_span(child_run)
        for child_run in [
            *run.child_llm_runs,
            *run.child_chain_runs,
            *run.child_tool_runs,
        ]
    ]
    base_span.span_kind = (
        trace_tree.SpanKind.AGENT
        # if isinstance(safely_get_span_producing_model(run), BaseSingleActionAgent)
        # else trace_tree.SpanKind.CHAIN
    )

    return base_span


def _convert_tool_run_to_wb_span(run: "ToolRun") -> "trace_tree.Span":
    base_span = _convert_run_to_wb_span(run)

    base_span.add_attribute("action", run.action)
    base_span.results = [
        trace_tree.Result(
            inputs={"input": run.tool_input}, outputs={"output": run.output}
        )
    ]
    base_span.child_spans = [
        _convert_lc_run_to_wb_span(child_run)
        for child_run in [
            *run.child_llm_runs,
            *run.child_chain_runs,
            *run.child_tool_runs,
        ]
    ]
    base_span.span_kind = trace_tree.SpanKind.TOOL

    return base_span


def _convert_run_to_wb_span(run: "BaseRun") -> "trace_tree.Span":
    attributes = {**run.extra} if run.extra else {}
    attributes["execution_order"] = run.execution_order

    return trace_tree.Span(
        span_id=str(run.uuid) if run.uuid is not None else None,
        name=run.serialized.get("name"),
        start_time_ms=int(run.start_time.timestamp() * 1000),
        end_time_ms=int(run.end_time.timestamp() * 1000),
        status_code=trace_tree.StatusCode.SUCCESS
        if run.error is None
        else trace_tree.StatusCode.ERROR,
        status_message=run.error,
        attributes=attributes,
    )


def _replace_type_with_kind(data: dict) -> dict:
    if isinstance(data, dict):
        # W&B TraceTree expects "_kind" instead of "_type" since `_type` is special
        # in W&B.
        if "_type" in data:
            _type = data.pop("_type")
            data["_kind"] = _type
        return {k: _replace_type_with_kind(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace_type_with_kind(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_replace_type_with_kind(v) for v in data)
    elif isinstance(data, set):
        return {_replace_type_with_kind(v) for v in data}
    else:
        return data


class WandbRunArgs(TypedDict):
    job_type: Optional[str]
    dir: Optional[StrPath]
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


class WandbTracer(BaseTracer):
    """Callback Handler that logs to Weights and Biases.

    This handler will log the model architecture and run traces to Weights and Biases. This will
    ensure that all LangChain activity is logged to W&B.
    """

    _run: Optional["WBRun"] = None
    _run_args: Optional[WandbRunArgs] = None

    def __init__(self, run_args: Optional[WandbRunArgs] = None, **kwargs: Any) -> None:
        """Initializes the WandbTracer.

        Parameters:
            run_args: (dict, optional) Arguments to pass to `wandb.init()`. If not provided, `wandb.init()` will be
                called with no arguments. Please refer to the `wandb.init` for more details.

        To use W&B to monitor all LangChain activity, add this tracer like any other langchain callback
        ```
        from wandb.integration.langchain import WandbTracer
        LLMChain(llm, callbacks=[WandbTracer()])
        # end of notebook / script:
        WandbTracer.finish()
        ```.
        """
        super().__init__(**kwargs)
        self._run_args = run_args
        self.session = self.load_session("")
        self._ensure_run(should_print_url=(wandb.run is None))

    @staticmethod
    def finish() -> None:
        """Waits for all asynchronous processes to finish and data to upload.

        Proxy for `wandb.finish()`.
        """
        wandb.finish()

    def _log_trace_from_run(self, run: "BaseRun") -> None:
        """Logs a LangChain Run to W*B as a W&B Trace."""
        self._ensure_run()

        root_span = safely_convert_lc_run_to_wb_span(run)
        if root_span is None:
            return

        model_dict = None

        # TODO: Uncomment this once we have a way to get the model from a run
        # model = safely_get_span_producing_model(run)
        # if model is not None:
        #     model_dict = safely_convert_model_to_dict(model)

        model_trace = trace_tree.WBTraceTree(
            root_span=root_span,
            model_dict=model_dict,
        )
        if wandb.run is not None:
            wandb.run.log({"langchain_trace": model_trace})

    def _ensure_run(self, should_print_url: bool = False) -> None:
        """Ensures an active W&B run exists.

        If not, will start a new run with the provided run_args.
        """
        if wandb.run is None:
            # Make a shallow copy of the run args, so we don't modify the original
            run_args = self._run_args or {}  # type: ignore
            run_args: dict = {**run_args}  # type: ignore

            # Prefer to run in silent mode since W&B has a lot of output
            # which can be undesirable when dealing with text-based models.
            if "settings" not in run_args:  # type: ignore
                run_args["settings"] = {"silent": True}  # type: ignore

            # Start the run and add the stream table
            wandb.init(**run_args)

            if should_print_url and wandb.run is not None:
                print_wandb_init_message(wandb.run.settings.run_url)

        with wb_telemetry.context(wandb.run) as tel:
            tel.feature.langchain_tracer = True

    # Start of required methods (these methods are required by the BaseCallbackHandler interface)
    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""
        return None

    def _persist_run(self, run: "BaseRun") -> None:
        """Persist a run."""
        try:
            self._log_trace_from_run(run)
        except Exception:
            # Silently ignore errors to not break user code
            pass

    def _persist_session(self, session_create: "TracerSessionBase") -> "TracerSession":
        """Persist a session."""
        try:
            return TracerSession(id=1, **session_create.dict())
        except Exception:
            return TracerSession(id=1)

    def load_session(self, session_name: str) -> "TracerSession":
        """Load a session from the tracer."""
        self._session = TracerSession(id=1)
        return self._session

    def load_default_session(self) -> "TracerSession":
        """Load the default tracing session and set it as the Tracer's session."""
        self._session = TracerSession(id=1)
        return self._session

    # End of required methods
