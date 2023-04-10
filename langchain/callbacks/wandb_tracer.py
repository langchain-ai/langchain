import json
import pathlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypedDict, Union

from langchain.callbacks import get_callback_manager
from langchain.callbacks.tracers.base import SharedTracer
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
    from langchain.callbacks.base import BaseCallbackHandler

    from wandb import Settings as WBSettings
    from wandb.integration.langchain.media_types import LangChainModelTrace
    from wandb.wandb_run import Run as WBRun


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
    def watch_all(cls, run_args: Optional[WandbRunArgs] = None, additional_handlers: list["BaseCallbackHandler"] =[]) -> "WandbTracer":
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
    def stop_watch():
        WandbTracer._instance.finish()
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

    def finish(self):
        """Waits for W&B data to upload. It is recommended to call this function
        before terminating the kernel or python script."""
        if self._run is not None:
            url = self._run.settings.run_url
            self._run.finish()
            import_wandb()
            from wandb.integration.langchain.media_types import print_wandb_finish_message
            print_wandb_finish_message(url)
        else:
            print("W&B run not started. Skipping.")

    @staticmethod
    def stop_all() -> None:
        """Stops all W&B runs."""
        global global_tracer
        if global_tracer is not None:
            global_tracer.stop()
            global_tracer = None

    def _log_trace(self, model_trace: "LangChainModelTrace") -> None:
        self._run.log({"_langchain_model_trace": model_trace})

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
        from wandb.integration.langchain.media_types import LangChainModelTrace

        model = run.serialized.get("_self")
        self._log_trace(LangChainModelTrace(run, model))

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
