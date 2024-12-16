"""A Tracer Implementation that records activity to Weights & Biases."""

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from langchain_core._api import warn_deprecated
from langchain_core.output_parsers.pydantic import PydanticBaseModel
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from wandb import Settings as WBSettings
    from wandb.sdk.data_types.trace_tree import Trace
    from wandb.sdk.lib.paths import StrPath
    from wandb.wandb_run import Run as WBRun

PRINT_WARNINGS = True


def _serialize_io(run_io: Optional[dict]) -> dict:
    """Utility to serialize the input and output of a run to store in wandb.
    Currently, supports serializing pydantic models and protobuf messages.

    :param run_io: The inputs and outputs of the run.
    :return: The serialized inputs and outputs.


    """
    if not run_io:
        return {}
    from google.protobuf.json_format import MessageToJson
    from google.protobuf.message import Message

    serialized_inputs = {}
    for key, value in run_io.items():
        if isinstance(value, Message):
            serialized_inputs[key] = MessageToJson(value)

        elif isinstance(value, PydanticBaseModel):
            serialized_inputs[key] = (
                value.model_dump_json()
                if hasattr(value, "model_dump_json")
                else value.json()
            )

        elif key == "input_documents":
            serialized_inputs.update(
                {f"input_document_{i}": doc.json() for i, doc in enumerate(value)}
            )
        else:
            serialized_inputs[key] = value
    return serialized_inputs


def flatten_run(run: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    runs: List[Dict[str, Any]], keep_keys: Tuple[str, ...] = ()
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

    def handle_id_and_kwargs(obj: Dict[str, Any], root: bool = False) -> Dict[str, Any]:
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
            if "data" in obj and isinstance(obj["data"], dict):
                obj = obj["data"]
            if ("id" in obj or "name" in obj) and not root:
                _kind = obj.get("id")
                if not _kind:
                    _kind = [obj.get("name")]
                if isinstance(_kind, list):
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

        if not name:
            name = _kind

        output_dict = {
            f"{name}": transformed_dict,
        }
        return output_dict

    return list(map(transform_run, runs))


def build_tree(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        parent_dict[next(iter(parent_dict))][next(iter(id_to_data[child_id]))] = (
            id_to_data[child_id][next(iter(id_to_data[child_id]))]
        )

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

    def __init__(
        self,
        run_args: Optional[WandbRunArgs] = None,
        io_serializer: Callable = _serialize_io,
        **kwargs: Any,
    ) -> None:
        """Initializes the WandbTracer.

        Parameters:
            run_args: (dict, optional) Arguments to pass to `wandb.init()`. If not
                provided, `wandb.init()` will be called with no arguments. Please
                refer to the `wandb.init` for more details.
            io_serializer: callable A function that serializes the input and outputs
             of a run to store in wandb. Defaults to "_serialize_io"

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
        self._io_serializer = io_serializer
        warn_deprecated(
            "0.3.8",
            pending=False,
            message=(
                "Please use the `WeaveTracer` from the `weave` package instead of this."
                "The `WeaveTracer` is a more flexible and powerful tool for logging "
                "and tracing your LangChain callables."
                "Find more information at https://weave-docs.wandb.ai/guides/integrations/langchain"
            ),
            alternative=(
                "Please instantiate the WeaveTracer from "
                "`weave.integrations.langchain import WeaveTracer` ."
                "For autologging simply use `weave.init()` and log all traces "
                "from your LangChain callables."
            ),
        )

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

    def process_model_dict(self, run: Run) -> Optional[Dict[str, Any]]:
        """Utility to process a run for wandb model_dict serialization.
        :param run: The run to process.
        :return: The convert model_dict to pass to WBTraceTree.
        """
        try:
            data = json.loads(run.json())
            processed = flatten_run(data)
            keep_keys = (
                "id",
                "name",
                "serialized",
                "parent_run_id",
            )
            processed = truncate_run_iterative(processed, keep_keys=keep_keys)
            exact_keys, partial_keys = (
                ("lc", "type", "graph"),
                (
                    "api_key",
                    "input",
                    "output",
                ),
            )
            processed = modify_serialized_iterative(
                processed, exact_keys=exact_keys, partial_keys=partial_keys
            )
            output = build_tree(processed)
            return output
        except Exception as e:
            if PRINT_WARNINGS:
                self._wandb.termerror(f"WARNING: Failed to serialize model: {e}")
            return None

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
                if run.run_type in ["llm", "tool"]:
                    run_type = run.run_type
                elif run.run_type == "chain":
                    run_type = "agent" if "agent" in run.name.lower() else "chain"
                else:
                    run_type = None

                metadata = get_metadata_dict(run)
                trace_tree = self._trace_tree.Trace(
                    name=run.name,
                    kind=run_type,
                    status_code="error" if run.error else "success",
                    start_time_ms=int(run.start_time.timestamp() * 1000)
                    if run.start_time is not None
                    else None,
                    end_time_ms=int(run.end_time.timestamp() * 1000)
                    if run.end_time is not None
                    else None,
                    metadata=metadata,
                    inputs=self._io_serializer(run.inputs),
                    outputs=self._io_serializer(run.outputs),
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
        model_dict = self.process_model_dict(run)
        if model_dict is not None and run_trace is not None:
            run_trace._model_dict = model_dict
        if self._wandb.run is not None and run_trace is not None:
            run_trace.log("langchain_trace")

    def _persist_run(self, run: "Run") -> None:
        """Persist a run."""
        self._log_trace_from_run(run)
