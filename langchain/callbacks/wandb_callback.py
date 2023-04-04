import json
import pathlib
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Sequence, TypedDict, Union

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import (
    ChainRun,
    LLMRun,
    ToolRun,
    TracerSession,
    TracerSessionCreate,
)
from langchain.llms import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.chains.base import Chain
from langchain.agents import Agent, AgentExecutor

if TYPE_CHECKING:
    from wandb.wandb_run import Run as WBRun
    from wandb import Settings as WBSettings
    from wandb.integration.langchain import LangChainTrace

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

class WandbCallbackHandler(BaseTracer):
    """Callback Handler that logs to Weights and Biases.

    Parameters:
        run_args (dict): The arguments to pass to wandb.init().

    This handler will log the model architecture and run traces to Weights and Biases.
    """
    _run: Optional["WBRun"] = None
    _run_args = Optional[WandbRunArgs] = None

    def __init__(
        self,
        run_args: dict # TODO: Type this
    ) -> None:
        """Initialize callback handler."""
        super().__init__()
        self._initialize_wandb_if_needed(run_args)

    def _initialize_wandb_if_needed(self, run_args: Optional[WandbRunArgs] = None) -> None:
        """Initialize wandb if it has not been initialized."""
        # We only want to start a new run if the run args differ. This will
        # reduce the number of W&B runs created, which is more ideal in a 
        # notebook setting
        if self._run is not None and self._run_args == run_args:
            return
        self._run_args = run_args

        # Make a shallow copy of the run args so we don't modify the original
        run_args = run_args or {}
        run_args = {**run_args}

        # Prefer to run in silent mode since W&B has a lot of output
        # which can be undesirable when dealing with text-based models.
        if 'settings' not in run_args:
            run_args['settings'] = {
                'silent': True
            }
        
        wandb = import_wandb()
        self._run = wandb.init(**run_args)
        
        warning = (
            "The wandb callback is currently in beta and is subject to change "
            "based on updates to `langchain`. Please report any issues to "
            "https://github.com/wandb/wandb/issues with the tag `langchain`."
        )
        wandb.termwarn(
            warning,
            repeat=False,
        )

        # self._run.printer()

    ###  Start of required methods
    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True
    
    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""
        return None
    
    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        print("Passing")
        # wandb = import_wandb()
        # StreamTable = wandb.sdk.data_types.stream_table.StreamTable
        # LangChainTrace = wandb.integration.langchain.LangChainTrace
        # try:
        #     self._log_trace(LangChainTrace(run))
        # except Exception as e:
        #     logging.warning(f"Failed to persist run: {e}")

    def _persist_session(self, session_create: TracerSessionCreate) -> TracerSession:
        """Persist a session."""
        return TracerSession(id=1, **session_create.dict())

    def load_session(self, session_name: str) -> TracerSession:
        """Load a session from the tracer."""
        self._session = TracerSession(id=1)
        return self._session

    def load_default_session(self) -> TracerSession:
        """Load the default tracing session and set it as the Tracer's session."""
        self._session = TracerSession(id=1)
        return self._session

    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_model = None
        self._traces_stream_table = None
        # self._models_stream_table = None
        self._run = None
        self._run_args = None

    def set_current_run_model(self, model: Union[BaseLLM, BaseChatModel, Chain, Agent, AgentExecutor], run_args: dict = {}):
        wandb = import_wandb()
        StreamTable = wandb.sdk.data_types.stream_table.StreamTable
        LangChainModel = wandb.integration.langchain.LangChainModel

        self._init_wandb_run(run_args)
        self._traces_stream_table = StreamTable(f"langchain_traces", ["model", "trace"])
        # self._models_stream_table = StreamTable(f"langchain_models", ["model"])
        if self._current_model is None or self._current_model._model != model:
            self._current_model = LangChainModel(model)
            self._save_model_to_wandb()
    
    # def _init_wandb_run(self, run_args: dict = {}):
    #     if self._run_args == None or json.dumps(run_args) != json.dumps(self._run_args):
    #         self._finish_wandb_run()
    #     if self._run is None:
    #         self._run_args = run_args
    #         self._run = wandb.init(**run_args)
    
    def _finish_wandb_run(self):
        if self._traces_stream_table is not None:
            self._traces_stream_table.join()
        # if self._models_stream_table is not None:
        #     self._models_stream_table.join()
        if self._run is not None:
            self._run.finish()
            self._run = None
        self._run_args = None

    def _log_trace(self, trace: LangChainTrace):
        if self._run is None:
            raise ValueError("No wandb run set for current run")
        
        self._traces_stream_table.add_data(self._current_model.model_id, self._current_model, trace)

    # def _save_model_to_wandb(self):
    #     if self._current_model is None:
    #         raise ValueError("No model set for current run")
    #     if self._run is None:
    #         raise ValueError("No wandb run set for current run")
    #     # Wrapping this in a table is a bit of a hack for now
    #     self._models_stream_table.add_data(self._current_model.model_id, self._current_model)
