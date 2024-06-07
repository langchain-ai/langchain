"""A Tracer Implementation that records activity to Weights & Biases."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypedDict, Union

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from wandb import Settings as WBSettings
    from wandb.sdk.data_types.trace_tree import Trace
    from wandb.sdk.lib.paths import StrPath
    from wandb.wandb_run import Run as WBRun

PRINT_WARNINGS = True


def _serialize_io(run_inputs: Optional[dict]) -> dict:
    if not run_inputs:
        return {}
    from google.protobuf.json_format import MessageToJson
    from google.protobuf.message import Message

    serialized_inputs = {}
    for key, value in run_inputs.items():
        if isinstance(value, Message):
            serialized_inputs[key] = MessageToJson(value)
        elif key == "input_documents":
            serialized_inputs.update(
                {f"input_document_{i}": doc.json() for i, doc in enumerate(value)}
            )
        else:
            serialized_inputs[key] = value
    return serialized_inputs


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

    def _ensure_run(self, should_print_url: bool = False) -> None:
        """Ensures an active W&B run exists.

        If not, will start a new run with the provided run_args.
        """
        if self._wandb.run is None:
            run_args: Dict = {**(self._run_args or {})}

            if "settings" not in run_args:
                run_args["settings"] = {"silent": True}

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

    def _log_trace_from_run(self, run: Run) -> None:
        """Logs a LangChain Run to W*B as a W&B Trace."""
        self._ensure_run()

        def create_trace(
            run: "Run", parent: Optional["Trace"] = None
        ) -> Optional["Trace"]:
            """
            Create a trace for a given run and its child runs.

            Args:
                run (Run): The run for which to create a trace.
                parent (Optional[Trace]): The parent trace.
                If provided, the created trace is added as a child to the parent trace.

            Returns:
                Optional[Trace]: The created trace.
                 If an error occurs during the creation of the trace, None is returned.

            Raises:
                Exception: If an error occurs during the creation of the trace,
                no exception is raised and a warning is printed.
            """

            def get_metadata_dict(r: "Run") -> Dict[str, Any]:
                """
                Extract metadata from a given run.

                This function extracts metadata from a given run
                and returns it as a dictionary.

                Args:
                    r (Run): The run from which to extract metadata.

                Returns:
                    Dict[str, Any]: A dictionary containing the extracted metadata.
                """
                run_dict = json.loads(r.json())
                metadata_dict = run_dict.get("metadata", {})
                metadata_dict["run_id"] = run_dict.get("id")
                metadata_dict["parent_run_id"] = run_dict.get("parent_run_id")
                metadata_dict["tags"] = run_dict.get("tags")
                metadata_dict["execution_order"] = run_dict.get(
                    "dotted_order", ""
                ).count(".")
                return metadata_dict

            try:
                trace_tree = self._trace_tree.Trace(
                    name=run.name,
                    kind=run.run_type
                    if run.run_type in ["llm", "chain", "tool"]
                    else None,
                    status_code="error" if run.error else "success",
                    start_time_ms=int(run.start_time.timestamp() * 1000)
                    if run.start_time is not None
                    else None,
                    end_time_ms=int(run.end_time.timestamp() * 1000)
                    if run.end_time is not None
                    else None,
                    metadata=get_metadata_dict(run),
                    inputs=_serialize_io(run.inputs),
                    outputs=_serialize_io(run.outputs),
                )

                # If the run has child runs, recursively create traces for them
                for child_run in run.child_runs:
                    create_trace(child_run, trace_tree)

                if parent is None:
                    return trace_tree
                else:
                    parent.add_child(trace_tree)
                    return parent
            except Exception as e:
                if PRINT_WARNINGS:
                    self._wandb.termwarn(
                        f"WARNING: Failed to serialize trace for run due to: {e}"
                    )
                return None

        run_trace = create_trace(run)
        if self._wandb.run is not None and run_trace is not None:
            run_trace.log("langchain_trace")

    def _persist_run(self, run: "Run") -> None:
        """Persist a run."""
        self._log_trace_from_run(run)
