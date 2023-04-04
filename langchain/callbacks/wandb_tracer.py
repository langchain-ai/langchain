import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Sequence, TypedDict, Union

import logging
import pathlib
import atexit

from langchain.callbacks import get_callback_manager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers.base import Tracer, SharedTracer
from langchain.callbacks.tracers.schemas import (
    ChainRun,
    LLMRun,
    TracerSession,
)
if TYPE_CHECKING:
    from langchain.callbacks.tracers.schemas import (
        ToolRun,
        TracerSessionCreate,
    )

if TYPE_CHECKING:
    from wandb.wandb_run import Run as WBRun
    from wandb import Settings as WBSettings
    from wandb.integration.langchain.media_types import LangChainTrace, LangChainModel
    from wandb.integration.langchain.stream_table import StreamTable

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

def watch(run_args: Optional[WandbRunArgs] = None):
    """Sets up a WandbTracer and makes it the default handler. To use W&B to 
    monitor all LangChain activity, simply call this function at the top of 
    the notebook or script:
    ```
    from langchain.callbacks.wandb_tracer import watch
    watch()
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
    tracer = WandbTracer()
    tracer.init(run_args)
    manager = get_callback_manager()
    manager.set_handlers([tracer, StdOutCallbackHandler()])

def finish():
    """Waits for W&B data to upload. It is recommended to call this function 
    before terminating the kernel or python script."""
    tracer = WandbTracer()
    if tracer._run is None:
        return
    url = tracer._run.settings.run_url
    tracer._run.finish()
    import_wandb().termlog((
        f"All files uploaded. View LangChain logs in W&B at {url}/chains."
    ))


def _print_wandb_url(run_url: str):
    import_wandb().termlog((
        f"W&B Run initialized. View LangChain logs in W&B at {run_url}/chains. "
        "To ensure that all data is uploaded, call `wandb_tracer.finish()` before "
        "terminating the notebook kernel or script."
        "\n\nNote that the WandbLangChainTracer is currently in beta and is subject to change "
        "based on updates to `langchain`. Please report any issues to "
        "https://github.com/wandb/wandb/issues with the tag `langchain`."
        )
    )

class WandbTracer(SharedTracer):
    """Callback Handler that logs to Weights and Biases.

    Parameters:
        run_args (dict): The arguments to pass to wandb.init().

    This handler will log the model architecture and run traces to Weights and Biases.
    """
    _run: Optional["WBRun"] = None
    _run_args: Optional[WandbRunArgs] = None
    _stream_table: Optional["StreamTable"] = None
    _known_models: dict[int, "LangChainModel"] = {}

    def init(self, run_args: Optional[WandbRunArgs] = None) -> None:
        """Initialize wandb if it has not been initialized."""
        # Load in wandb symbols
        wandb = import_wandb()
        from wandb.integration.langchain.stream_table import StreamTable
        from wandb.sdk.wandb_run import TeardownHook, TeardownStage

        # We only want to start a new run if the run args differ. This will
        # reduce the number of W&B runs created, which is more ideal in a 
        # notebook setting
        if wandb.run != None and self._run is not None and json.dumps(self._run_args, sort_keys=True) == json.dumps(run_args, sort_keys=True):
            _print_wandb_url(self._run.settings.run_url)
            return
        self._run_args = run_args
        self._run = None
        self._stream_table = None

        # Make a shallow copy of the run args so we don't modify the original
        run_args = run_args or {}
        run_args = {**run_args}

        # Prefer to run in silent mode since W&B has a lot of output
        # which can be undesirable when dealing with text-based models.
        if 'settings' not in run_args:
            run_args['settings'] = {
                'silent': True
            }

        # Start the run and add the stream table
        self._run = wandb.init(**run_args)
        self._stream_table = StreamTable(f"langchain_traces", ["model", "trace"])
        self._run._teardown_hooks.append(TeardownHook(self._on_run_teardown, TeardownStage.EARLY))
        atexit.register(self._on_run_teardown)

        # Call this for the user since we only use a single session id in the tracer at the moment
        self.load_session("")

        _print_wandb_url(self._run.settings.run_url)
        

    def _on_run_teardown(self):
        if self._stream_table is not None:
            self._stream_table.join()
        self._run = None
        self._run_args = None
        self._stream_table = None

    def _log_trace(self, model: "LangChainModel", trace: "LangChainTrace"):
        if self._stream_table is None:
            logging.warning("Failed to log trace to W&B. No StreamTable found.")
        
        # self._stream_table.add_data(self._current_model, trace)
        self._stream_table.add_data(model, trace)


    ###  Start of required methods
    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True
    
    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""
        return None
    
    def _persist_run(self, run: Union["LLMRun", "ChainRun", "ToolRun"]) -> None:
        """Persist a run."""
        import_wandb()
        from wandb.integration.langchain.media_types import LangChainTrace, LangChainModel
        try:
            wb_model = None
            model = run.serialized.get('_self')
            if model is not None:
                key = id(model)
                # warning: map of models is not thread safe and has unbounded memory usage
                if key not in self._known_models or self._known_models[key]._model != model:
                    self._known_models[key] = LangChainModel(model)
                wb_model = self._known_models[key]
            self._log_trace(wb_model, LangChainTrace(run))
        except Exception as e:
            raise e
            # logging.warning(f"Failed to persist run: {e}")

    def _persist_session(self, session_create: "TracerSessionCreate") -> "TracerSession":
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
        if isinstance(child_run, LLMRun):
            parent_run.child_llm_runs.append(child_run)
        elif isinstance(child_run, ChainRun):
            parent_run.child_chain_runs.append(child_run)
        else:
            parent_run.child_tool_runs.append(child_run)
        parent_run.child_runs.append(child_run)
    ## End of required methods
