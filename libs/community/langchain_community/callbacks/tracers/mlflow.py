"""A Tracer Implementation that records activity to MLflow."""
import logging
import os
import random
import string
import uuid
from typing import Any, Dict, Optional

from langchain_core.env import get_runtime_environment
from langchain_community.callbacks.utils import (
    flatten_dict,
)
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)


def import_mlflow() -> Any:
    """Import the mlflow python package and raise an error if it is not installed."""
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "To use MLflowTracer you need to have the `mlflow` python "
            "package installed. Please install it with `pip install mlflow -U`"
        )
    return mlflow


class MLflowTracer(BaseTracer):
    """Callback Handler that logs to MLflow.

    This handler will log the model architecture and run traces to MLflow.
    This will ensure that all LangChain activity is logged to MLflow.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.mlflow = import_mlflow()
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            self.mlflow.set_tracking_uri("databricks")
            self.mlf_expid = self.mlflow.tracking.fluent._get_experiment_id()
        else:
            if tracking_uri:
                self.mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                if exp := self.mlflow.get_experiment_by_name(experiment_name):
                    self.mlf_expid = exp.experiment_id
                else:
                    self.mlf_expid = self.mlflow.create_experiment(experiment_name)

        self.tags = tags or {}
        if run_id is None:
            if run_name is None:
                run_name = "langchain-tracer-" + "".join(
                    random.choices(string.ascii_uppercase + string.digits, k=7)
                )
            run = self.mlflow.MlflowClient().create_run(
                self.mlf_expid, run_name=run_name, tags=tags
            )
            self.run_id = run.info.run_id
        else:
            self.run_id = run_id
        self.session_id = uuid.uuid4().hex
        self.run_dict = {}
        self.run_table = "langchain_runs.json"

    def _convert_type(self, value: Any) -> Any:
        """Convert a value to a type that can be json-serialized."""
        if isinstance(value, dict):
            for k, v in value.items():
                value[k] = self._convert_type(v)
        elif isinstance(value, list):
            value = [self._convert_type(v) for v in value]
        elif not isinstance(value, (str, int, float, bool, type(None))):
            value = str(value)
        return value

    def _convert_run_to_dict(self, run: Run) -> dict:
        """Convert a Run object to a dictionary."""
        run_dict = run.dict(exclude={"child_runs"})
        extra = run_dict.get("extra", {})
        extra["runtime"] = get_runtime_environment()
        run_dict["extra"] = extra
        run_dict["session_id"] = self.session_id
        run_dict["child_runs"] = [str(run.id) for run in run.child_runs]
        if run.serialized:
            run_dict["serialized_object"] = flatten_dict(run.serialized)
        run_dict = self._convert_type(run_dict)
        return run_dict

    def _log_trace_from_run(self, run_dict: Dict[str, Any]):
        """Log the trace of a run dictionary into MLflow."""
        self.mlflow.log_table(run_dict, self.run_table, self.run_id)
        for child_run_id in run_dict.get("child_runs", []):
            if child_run_id not in self.run_dict:
                logger.debug(
                    f"Child run {child_run_id} not found in run_dict {self.run_dict}"
                )
            else:
                self._log_trace_from_run(self.run_dict[child_run_id])

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        # Only persist the runs without a parent
        run_dict = self._convert_run_to_dict(run)
        self.run_dict[str(run.id)] = run_dict
        self._log_trace_from_run(run_dict)

    def _on_run_update(self, run: Run) -> None:
        """Process a run upon update."""
        # save the run when the run trace ends
        if run.parent_run_id:
            self.run_dict[str(run.id)] = self._convert_run_to_dict(run)

    def end_run(self):
        """End the run."""
        self.mlflow.MlflowClient().set_terminated(self.run_id)
