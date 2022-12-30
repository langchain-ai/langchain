"""An implementation of the Tracer interface that POSTS to the langchain endpoint."""

from typing import Optional, Union

import requests

from langchain.callbacks.tracing.tracer import Tracer, LLMRun, ChainRun, ToolRun


class LangChainTracer(Tracer):
    """An implementation of the Tracer interface that POSTS to the langchain endpoint."""

    _endpoint: str = "http://127.0.0.1:5000"

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

        if isinstance(run, LLMRun):
            endpoint = f"{self._endpoint}/llm-runs"
        elif isinstance(run, ChainRun):
            endpoint = f"{self._endpoint}/chain-runs"
        else:
            endpoint = f"{self._endpoint}/tool-runs"
        r = requests.post(
            endpoint,
            data=run.to_json(),
            headers={"Content-Type": "application/json"},
        )
        print(f"POST {endpoint}, status code: {r.status_code}, id: {r.json()['id']}")

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

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""

        return None
