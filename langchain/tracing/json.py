"""An implementation of the Tracer interface that prints trace as nested json."""

import uuid
from typing import Optional, Union

from langchain.tracing.base import ChainRun, LLMRun, ToolRun
from langchain.tracing.nested import NestedTracer


class JsonTracer(NestedTracer):
    """An implementation of the Tracer interface that prints trace as nested json."""

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

        print(run.to_json(indent=2))

    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""

        parent_run.child_runs.append(child_run)

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""

        return str(uuid.uuid4())
