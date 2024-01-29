"""A Tracer Implementation that records activity to Weights & Biases."""
from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from wandb import Settings as WBSettings
    from wandb.sdk.data_types.trace_tree import Span
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


class RunProcessor:
    """Handles the conversion of a LangChain Runs into a WBTraceTree."""

    def __init__(self, wandb_module: Any, trace_module: Any):
        self.wandb = wandb_module
        self.trace_tree = trace_module

    def process_span(self, run: Run) -> Optional["Span"]:
        """Converts a LangChain Run into a W&B Trace Span.
        :param run: The LangChain Run to convert.
        :return: The converted W&B Trace Span.
        """
        try:
            span = self._convert_lc_run_to_wb_span(run)
            return span
        except Exception as e:
            if PRINT_WARNINGS:
                self.wandb.termwarn(
                    f"Skipping trace saving - unable to safely convert LangChain Run "
                    f"into W&B Trace due to: {e}"
                )
            return None

    def _convert_run_to_wb_span(self, run: Run) -> "Span":
        """Base utility to create a span from a run.
        :param run: The run to convert.
        :return: The converted Span.
        """
        attributes = {**run.extra} if run.extra else {}
        attributes["execution_order"] = run.execution_order

        return self.trace_tree.Span(
            span_id=str(run.id) if run.id is not None else None,
            name=run.name,
            start_time_ms=int(run.start_time.timestamp() * 1000),
            end_time_ms=int(run.end_time.timestamp() * 1000)
            if run.end_time is not None
            else None,
            status_code=self.trace_tree.StatusCode.SUCCESS
            if run.error is None
            else self.trace_tree.StatusCode.ERROR,
            status_message=run.error,
            attributes=attributes,
        )

    def _convert_llm_run_to_wb_span(self, run: Run) -> "Span":
        """Converts a LangChain LLM Run into a W&B Trace Span.
        :param run: The LangChain LLM Run to convert.
        :return: The converted W&B Trace Span.
        """
        base_span = self._convert_run_to_wb_span(run)
        if base_span.attributes is None:
            base_span.attributes = {}
        base_span.attributes["llm_output"] = (run.outputs or {}).get("llm_output", {})

        base_span.results = [
            self.trace_tree.Result(
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
        base_span.span_kind = self.trace_tree.SpanKind.LLM

        return base_span

    def _convert_chain_run_to_wb_span(self, run: Run) -> "Span":
        """Converts a LangChain Chain Run into a W&B Trace Span.
        :param run: The LangChain Chain Run to convert.
        :return: The converted W&B Trace Span.
        """
        base_span = self._convert_run_to_wb_span(run)

        base_span.results = [
            self.trace_tree.Result(
                inputs=_serialize_io(run.inputs), outputs=_serialize_io(run.outputs)
            )
        ]
        base_span.child_spans = [
            self._convert_lc_run_to_wb_span(child_run) for child_run in run.child_runs
        ]
        base_span.span_kind = (
            self.trace_tree.SpanKind.AGENT
            if "agent" in run.name.lower()
            else self.trace_tree.SpanKind.CHAIN
        )

        return base_span

    def _convert_tool_run_to_wb_span(self, run: Run) -> "Span":
        """Converts a LangChain Tool Run into a W&B Trace Span.
        :param run: The LangChain Tool Run to convert.
        :return: The converted W&B Trace Span.
        """
        base_span = self._convert_run_to_wb_span(run)
        base_span.results = [
            self.trace_tree.Result(
                inputs=_serialize_io(run.inputs), outputs=_serialize_io(run.outputs)
            )
        ]
        base_span.child_spans = [
            self._convert_lc_run_to_wb_span(child_run) for child_run in run.child_runs
        ]
        base_span.span_kind = self.trace_tree.SpanKind.TOOL

        return base_span

    def _convert_lc_run_to_wb_span(self, run: Run) -> "Span":
        """Utility to convert any generic LangChain Run into a W&B Trace Span.
        :param run: The LangChain Run to convert.
        :return: The converted W&B Trace Span.
        """
        if run.run_type == "llm":
            return self._convert_llm_run_to_wb_span(run)
        elif run.run_type == "chain":
            return self._convert_chain_run_to_wb_span(run)
        elif run.run_type == "tool":
            return self._convert_tool_run_to_wb_span(run)
        else:
            return self._convert_run_to_wb_span(run)

    def process_model(self, run: Run) -> Optional[Dict[str, Any]]:
        """Utility to process a run for wandb model_dict serialization.
        :param run: The run to process.
        :return: The convert model_dict to pass to WBTraceTree.
        """
        try:
            data = json.loads(run.json())
            processed = self.flatten_run(data)
            keep_keys = (
                "id",
                "name",
                "serialized",
                "inputs",
                "outputs",
                "parent_run_id",
                "execution_order",
            )
            processed = self.truncate_run_iterative(processed, keep_keys=keep_keys)
            exact_keys, partial_keys = ("lc", "type"), ("api_key",)
            processed = self.modify_serialized_iterative(
                processed, exact_keys=exact_keys, partial_keys=partial_keys
            )
            output = self.build_tree(processed)
            return output
        except Exception as e:
            if PRINT_WARNINGS:
                self.wandb.termwarn(f"WARNING: Failed to serialize model: {e}")
            return None

    def flatten_run(self, run: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Utility to flatten a nest run object into a list of runs.
        :param run: The base run to flatten.
        :return: The flattened list of runs.
        """

        def flatten(child_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Utility to recursively flatten a list of child runs in a run.
            :param child_runs: The list of child runs to flatten.
            :return: The flattened list of runs.
            """
            if child_runs is None:
                return []

            result = []
            for item in child_runs:
                child_runs = item.pop("child_runs", [])
                result.append(item)
                result.extend(flatten(child_runs))

            return result

        return flatten([run])

    def truncate_run_iterative(
        self, runs: List[Dict[str, Any]], keep_keys: Tuple[str, ...] = ()
    ) -> List[Dict[str, Any]]:
        """Utility to truncate a list of runs dictionaries to only keep the specified
            keys in each run.
        :param runs: The list of runs to truncate.
        :param keep_keys: The keys to keep in each run.
        :return: The truncated list of runs.
        """

        def truncate_single(run: Dict[str, Any]) -> Dict[str, Any]:
            """Utility to truncate a single run dictionary to only keep the specified
                keys.
            :param run: The run dictionary to truncate.
            :return: The truncated run dictionary
            """
            new_dict = {}
            for key in run:
                if key in keep_keys:
                    new_dict[key] = run.get(key)
            return new_dict

        return list(map(truncate_single, runs))

    def modify_serialized_iterative(
        self,
        runs: List[Dict[str, Any]],
        exact_keys: Tuple[str, ...] = (),
        partial_keys: Tuple[str, ...] = (),
    ) -> List[Dict[str, Any]]:
        """Utility to modify the serialized field of a list of runs dictionaries.
        removes any keys that match the exact_keys and any keys that contain any of the
        partial_keys.
        recursively moves the dictionaries under the kwargs key to the top level.
        changes the "id" field to a string "_kind" field that tells WBTraceTree how to
        visualize the run. promotes the "serialized" field to the top level.

        :param runs: The list of runs to modify.
        :param exact_keys: A tuple of keys to remove from the serialized field.
        :param partial_keys: A tuple of partial keys to remove from the serialized
            field.
        :return: The modified list of runs.
        """

        def remove_exact_and_partial_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively removes exact and partial keys from a dictionary.
            :param obj: The dictionary to remove keys from.
            :return: The modified dictionary.
            """
            if isinstance(obj, dict):
                obj = {
                    k: v
                    for k, v in obj.items()
                    if k not in exact_keys
                    and not any(partial in k for partial in partial_keys)
                }
                for k, v in obj.items():
                    obj[k] = remove_exact_and_partial_keys(v)
            elif isinstance(obj, list):
                obj = [remove_exact_and_partial_keys(x) for x in obj]
            return obj

        def handle_id_and_kwargs(
            obj: Dict[str, Any], root: bool = False
        ) -> Dict[str, Any]:
            """Recursively handles the id and kwargs fields of a dictionary.
            changes the id field to a string "_kind" field that tells WBTraceTree how
            to visualize the run. recursively moves the dictionaries under the kwargs
            key to the top level.
            :param obj: a run dictionary with id and kwargs fields.
            :param root: whether this is the root dictionary or the serialized
                dictionary.
            :return: The modified dictionary.
            """
            if isinstance(obj, dict):
                if ("id" in obj or "name" in obj) and not root:
                    _kind = obj.get("id")
                    if not _kind:
                        _kind = [obj.get("name")]
                    obj["_kind"] = _kind[-1]
                    obj.pop("id", None)
                    obj.pop("name", None)
                    if "kwargs" in obj:
                        kwargs = obj.pop("kwargs")
                        for k, v in kwargs.items():
                            obj[k] = v
                for k, v in obj.items():
                    obj[k] = handle_id_and_kwargs(v)
            elif isinstance(obj, list):
                obj = [handle_id_and_kwargs(x) for x in obj]
            return obj

        def transform_serialized(serialized: Dict[str, Any]) -> Dict[str, Any]:
            """Transforms the serialized field of a run dictionary to be compatible
                with WBTraceTree.
            :param serialized: The serialized field of a run dictionary.
            :return: The transformed serialized field.
            """
            serialized = handle_id_and_kwargs(serialized, root=True)
            serialized = remove_exact_and_partial_keys(serialized)
            return serialized

        def transform_run(run: Dict[str, Any]) -> Dict[str, Any]:
            """Transforms a run dictionary to be compatible with WBTraceTree.
            :param run: The run dictionary to transform.
            :return: The transformed run dictionary.
            """
            transformed_dict = transform_serialized(run)

            serialized = transformed_dict.pop("serialized")
            for k, v in serialized.items():
                transformed_dict[k] = v

            _kind = transformed_dict.get("_kind", None)
            name = transformed_dict.pop("name", None)
            exec_ord = transformed_dict.pop("execution_order", None)

            if not name:
                name = _kind

            output_dict = {
                f"{exec_ord}_{name}": transformed_dict,
            }
            return output_dict

        return list(map(transform_run, runs))

    def build_tree(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Builds a nested dictionary from a list of runs.
        :param runs: The list of runs to build the tree from.
        :return: The nested dictionary representing the langchain Run in a tree
            structure compatible with WBTraceTree.
        """
        id_to_data = {}
        child_to_parent = {}

        for entity in runs:
            for key, data in entity.items():
                id_val = data.pop("id", None)
                parent_run_id = data.pop("parent_run_id", None)
                id_to_data[id_val] = {key: data}
                if parent_run_id:
                    child_to_parent[id_val] = parent_run_id

        for child_id, parent_id in child_to_parent.items():
            parent_dict = id_to_data[parent_id]
            parent_dict[next(iter(parent_dict))][
                next(iter(id_to_data[child_id]))
            ] = id_to_data[child_id][next(iter(id_to_data[child_id]))]

        root_dict = next(
            data for id_val, data in id_to_data.items() if id_val not in child_to_parent
        )

        return root_dict


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
        self.run_processor = RunProcessor(self._wandb, self._trace_tree)

    def finish(self) -> None:
        """Waits for all asynchronous processes to finish and data to upload.

        Proxy for `wandb.finish()`.
        """
        self._wandb.finish()

    def _log_trace_from_run(self, run: Run) -> None:
        """Logs a LangChain Run to W*B as a W&B Trace."""
        self._ensure_run()

        root_span = self.run_processor.process_span(run)
        model_dict = self.run_processor.process_model(run)

        if root_span is None:
            return

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
            run_args = self._run_args or {}  # type: ignore
            run_args: dict = {**run_args}  # type: ignore

            if "settings" not in run_args:  # type: ignore
                run_args["settings"] = {"silent": True}  # type: ignore

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
