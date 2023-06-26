"""A Tracer Implementation that records activity to Weights & Biases."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run, RunTypeEnum

if TYPE_CHECKING:
    from wandb import Settings as WBSettings
    from wandb.sdk.data_types import trace_tree
    from wandb.sdk.lib.paths import StrPath
    from wandb.wandb_run import Run as WBRun


PRINT_WARNINGS = True


def _convert_lc_run_to_wb_span(trace_tree: Any, run: Run) -> trace_tree.Span:
    if run.run_type == RunTypeEnum.llm:
        return _convert_llm_run_to_wb_span(trace_tree, run)
    elif run.run_type == RunTypeEnum.chain:
        return _convert_chain_run_to_wb_span(trace_tree, run)
    elif run.run_type == RunTypeEnum.tool:
        return _convert_tool_run_to_wb_span(trace_tree, run)
    else:
        return _convert_run_to_wb_span(trace_tree, run)


def _convert_llm_run_to_wb_span(trace_tree: Any, run: Run) -> trace_tree.Span:
    base_span = _convert_run_to_wb_span(trace_tree, run)

    base_span.results = [
        trace_tree.Result(
            inputs={"prompt": prompt},
            outputs={
                f"gen_{g_i}": gen["text"]
                for g_i, gen in enumerate(run.outputs["generations"][ndx])
            }
            if (
                run.outputs is not None
                and len(run.outputs["generations"]) > ndx
                and len(run.outputs["generations"][ndx]) > 0
            )
            else None,
        )
        for ndx, prompt in enumerate(run.inputs["prompts"] or [])
    ]
    base_span.span_kind = trace_tree.SpanKind.LLM

    return base_span


def _serialize_inputs(run_inputs: dict) -> Union[dict, list]:
    if "input_documents" in run_inputs:
        docs = run_inputs["input_documents"]
        return [doc.json() for doc in docs]
    else:
        return run_inputs


def _convert_chain_run_to_wb_span(trace_tree: Any, run: Run) -> trace_tree.Span:
    base_span = _convert_run_to_wb_span(trace_tree, run)

    base_span.results = [
        trace_tree.Result(inputs=_serialize_inputs(run.inputs), outputs=run.outputs)
    ]
    base_span.child_spans = [
        _convert_lc_run_to_wb_span(trace_tree, child_run)
        for child_run in run.child_runs
    ]
    base_span.span_kind = (
        trace_tree.SpanKind.AGENT
        if "agent" in run.serialized.get("name", "").lower()
        else trace_tree.SpanKind.CHAIN
    )

    return base_span


def _convert_tool_run_to_wb_span(trace_tree: Any, run: Run) -> trace_tree.Span:
    base_span = _convert_run_to_wb_span(trace_tree, run)
    base_span.results = [
        trace_tree.Result(inputs=_serialize_inputs(run.inputs), outputs=run.outputs)
    ]
    base_span.child_spans = [
        _convert_lc_run_to_wb_span(trace_tree, child_run)
        for child_run in run.child_runs
    ]
    base_span.span_kind = trace_tree.SpanKind.TOOL

    return base_span


def _convert_run_to_wb_span(trace_tree: Any, run: Run) -> trace_tree.Span:
    attributes = {**run.extra} if run.extra else {}
    attributes["execution_order"] = run.execution_order

    return trace_tree.Span(
        span_id=str(run.id) if run.id is not None else None,
        name=run.serialized.get("name"),
        start_time_ms=int(run.start_time.timestamp() * 1000),
        end_time_ms=int(run.end_time.timestamp() * 1000),
        status_code=trace_tree.StatusCode.SUCCESS
        if run.error is None
        else trace_tree.StatusCode.ERROR,
        status_message=run.error,
        attributes=attributes,
    )


def _replace_type_with_kind(data: Any) -> Any:
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
    """Arguments for the WandbTracer."""

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
    settings: Union[WBSettings, Dict[str, Any], None]


class WandbTracer(BaseTracer):
    """Callback Handler that logs to Weights and Biases.

    This handler will log the model architecture and run traces to Weights and Biases.
    This will ensure that all LangChain activity is logged to W&B.
    """

    _run: Optional[WBRun] = None
    _run_args: Optional[WandbRunArgs] = None

    def __init__(self, run_args: Optional[WandbRunArgs] = None, **kwargs: Any) -> None:
        """Initializes the WandbTracer.

        Parameters:
            run_args: (dict, optional) Arguments to pass to `wandb.init()`. If not
                provided, `wandb.init()` will be called with no arguments. Please
                refer to the `wandb.init` for more details.

        To use W&B to monitor all LangChain activity, add this tracer like any other
        LangChain callback:
        ```
        from wandb.integration.langchain import WandbTracer

        tracer = WandbTracer()
        chain = LLMChain(llm, callbacks=[tracer])
        # ...end of notebook / script:
        tracer.finish()
        ```
        """
        super().__init__(**kwargs)
        try:
            import wandb
            from wandb.sdk.data_types import trace_tree
        except ImportError as e:
            raise ImportError(
                "Could not import wandb python package."
                "Please install it with `pip install wandb`."
            ) from e
        self._wandb = wandb
        self._trace_tree = trace_tree
        self._run_args = run_args
        self._ensure_run(should_print_url=(wandb.run is None))

    def finish(self) -> None:
        """Waits for all asynchronous processes to finish and data to upload.

        Proxy for `wandb.finish()`.
        """
        self._wandb.finish()

    def _log_trace_from_run(self, run: Run) -> None:
        """Logs a LangChain Run to W*B as a W&B Trace."""
        self._ensure_run()

        try:
            root_span = _convert_lc_run_to_wb_span(self._trace_tree, run)
        except Exception as e:
            if PRINT_WARNINGS:
                self._wandb.termwarn(
                    f"Skipping trace saving - unable to safely convert LangChain Run "
                    f"into W&B Trace due to: {e}"
                )
            return

        model_dict = None

        # TODO: Add something like this once we have a way to get the clean serialized
        # parent dict from a run:
        # serialized_parent = safely_get_span_producing_model(run)
        # if serialized_parent is not None:
        #   model_dict = safely_convert_model_to_dict(serialized_parent)

        model_trace = self._trace_tree.WBTraceTree(
            root_span=root_span,
            model_dict=model_dict,
        )
        if self._wandb.run is not None:
            self._wandb.run.log({"langchain_trace": model_trace})

    def _ensure_run(self, should_print_url: bool = False) -> None:
        """Ensures an active W&B run exists.

        If not, will start a new run with the provided run_args.
        """
        if self._wandb.run is None:
            # Make a shallow copy of the run args, so we don't modify the original
            run_args = self._run_args or {}  # type: ignore
            run_args: dict = {**run_args}  # type: ignore

            # Prefer to run in silent mode since W&B has a lot of output
            # which can be undesirable when dealing with text-based models.
            if "settings" not in run_args:  # type: ignore
                run_args["settings"] = {"silent": True}  # type: ignore

            # Start the run and add the stream table
            self._wandb.init(**run_args)
            if self._wandb.run is not None:
                if should_print_url:
                    run_url = self._wandb.run.settings.run_url
                    self._wandb.termlog(
                        f"Streaming LangChain activity to W&B at {run_url}\n"
                        "`WandbTracer` is currently in beta.\n"
                        "Please report any issues to "
                        "https://github.com/wandb/wandb/issues with the tag "
                        "`langchain`."
                    )

                self._wandb.run._label(repo="langchain")

    def _persist_run(self, run: "Run") -> None:
        """Persist a run."""
        self._log_trace_from_run(run)
