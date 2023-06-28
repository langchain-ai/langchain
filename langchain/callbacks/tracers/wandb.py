"""A Tracer Implementation that records activity to Weights & Biases."""
from __future__ import annotations

import copy
import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypedDict, Union

import langchain
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run, RunTypeEnum

if TYPE_CHECKING:
    from wandb import Settings as WBSettings
    from wandb.sdk.data_types import trace_tree
    from wandb.sdk.lib.paths import StrPath
    from wandb.wandb_run import Run as WBRun


PRINT_WARNINGS = True


def _serialize_inputs(run_inputs: dict) -> Union[dict, list]:
    if "input_documents" in run_inputs:
        docs = run_inputs["input_documents"]
        return [doc.json() for doc in docs]
    else:
        return run_inputs


def _maybe_replace_key(
    obj: Dict[str, Any],
    key: str,
) -> None:
    if key in obj and isinstance(obj[key], list):
        obj[key] = ".".join(obj[key])

    for k, v in obj.items():
        if isinstance(v, dict):
            _maybe_replace_key(v, key)


def _maybe_scrub_key(obj: Dict[str, Any], key: str) -> None:
    if isinstance(obj, dict):
        # the call to `list` is useless for py2 but makes
        # the code py2/py3 compatible
        for k in list(obj.keys()):
            if k == key or "api_key" in k.lower():
                del obj[k]
            else:
                _maybe_scrub_key(obj[k], key)
    elif isinstance(obj, list):
        for i in reversed(range(len(obj))):
            if obj[i] == key or "api_key" in obj[i].lower():
                del obj[i]
            else:
                _maybe_scrub_key(obj[i], key)

    else:
        # neither a dict nor a list, do nothing
        pass


def _load_class_from_name(module_name: str, class_name: str) -> Any:
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def _maybe_infer_kind_from_id(id_value: str) -> str:
    id_value_splits = id_value.rsplit(".", 1)
    if len(id_value_splits) == 2:
        module_name, class_name = id_value_splits
        id_value = _load_class_from_name(module_name, class_name)
    else:
        return "unknown"
    if issubclass(id_value, langchain.agents.agent.AgentExecutor):
        return "agentexecutor"
    elif issubclass(id_value, langchain.chains.base.Chain):
        return "llm_chain"
    elif issubclass(id_value, langchain.llms.base.BaseLLM):
        return "llm"
    elif issubclass(id_value, langchain.prompts.prompt.PromptTemplate):
        return "prompt"
    else:
        return "unknown"


def _identify_kind_and_add_key(d: Dict[str, Any]) -> Any:
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                if "id" in value:
                    value["_kind"] = _maybe_infer_kind_from_id(value["id"])
                elif "name" in value:
                    value["_kind"] = value["name"]
                _identify_kind_and_add_key(value)  # Recursive call for dictionary value
            elif isinstance(value, list):
                for item in value:
                    _identify_kind_and_add_key(
                        item
                    )  # Recursive call for each item in the list
    elif isinstance(d, list):
        for item in d:
            _identify_kind_and_add_key(item)  # Recursive call for each item in the list


def _safely_convert_model_to_dict(run: Run) -> Dict[str, Any]:
    try:
        serialized = copy.deepcopy(run.serialized)
        serialized["_kind"] = run.run_type.value

        model_dict = {f"{run.execution_order}_{run.name}": serialized}

        for child_run in run.child_runs:
            serialized = copy.deepcopy(child_run.serialized)
            serialized["_kind"] = child_run.run_type.value
            model_dict[f"{child_run.execution_order}_{child_run.name}"] = serialized

        _maybe_replace_key(model_dict, "id")
        for key in ("lc", "type"):
            _maybe_scrub_key(model_dict, key)

        _identify_kind_and_add_key(model_dict)

        return model_dict
    except Exception as e:
        if PRINT_WARNINGS:
            print(f"WARNING: Failed to serialize model: {e}")
        return {}


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
        if "agent" in run.name.lower()
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


def _convert_lc_run_to_wb_span(trace_tree: Any, run: Run) -> trace_tree.Span:
    if run.run_type == RunTypeEnum.llm:
        return _convert_llm_run_to_wb_span(trace_tree, run)
    elif run.run_type == RunTypeEnum.chain:
        return _convert_chain_run_to_wb_span(trace_tree, run)
    elif run.run_type == RunTypeEnum.tool:
        return _convert_tool_run_to_wb_span(trace_tree, run)
    else:
        return _convert_run_to_wb_span(trace_tree, run)


def _convert_run_to_wb_span(trace_tree: Any, run: Run) -> trace_tree.Span:
    attributes = {**run.extra} if run.extra else {}
    attributes["execution_order"] = run.execution_order

    return trace_tree.Span(
        span_id=str(run.id) if run.id is not None else None,
        name=run.name,
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
                "Please install it with `pip install -U wandb`."
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

        model_dict = _safely_convert_model_to_dict(run)

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
