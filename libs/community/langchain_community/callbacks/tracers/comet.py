from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict

from langchain_core.tracers import BaseTracer
from langchain_core.utils import guard_import

if TYPE_CHECKING:
    from uuid import UUID

    from comet_llm import Span
    from comet_llm.chains.chain import Chain

    from langchain_community.callbacks.tracers.schemas import Run


def _get_run_type(run: "Run") -> str:
    if isinstance(run.run_type, str):
        return run.run_type
    elif hasattr(run.run_type, "value"):
        return run.run_type.value
    else:
        return str(run.run_type)


def import_comet_llm_api() -> SimpleNamespace:
    """Import comet_llm api and raise an error if it is not installed."""
    comet_llm = guard_import("comet_llm")
    comet_llm_chains = guard_import("comet_llm.chains")

    return SimpleNamespace(
        chain=comet_llm_chains.chain,
        span=comet_llm_chains.span,
        chain_api=comet_llm_chains.api,
        experiment_info=comet_llm.experiment_info,
        flush=comet_llm.flush,
    )


class CometTracer(BaseTracer):
    """Comet Tracer."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Comet Tracer."""
        super().__init__(**kwargs)
        self._span_map: Dict["UUID", "Span"] = {}
        """Map from run id to span."""
        self._chains_map: Dict["UUID", "Chain"] = {}
        """Map from run id to chain."""
        self._initialize_comet_modules()

    def _initialize_comet_modules(self) -> None:
        comet_llm_api = import_comet_llm_api()
        self._chain: ModuleType = comet_llm_api.chain
        self._span: ModuleType = comet_llm_api.span
        self._chain_api: ModuleType = comet_llm_api.chain_api
        self._experiment_info: ModuleType = comet_llm_api.experiment_info
        self._flush: Callable[[], None] = comet_llm_api.flush

    def _persist_run(self, run: "Run") -> None:
        run_dict: Dict[str, Any] = run.dict()
        chain_ = self._chains_map[run.id]
        chain_.set_outputs(outputs=run_dict["outputs"])
        self._chain_api.log_chain(chain_)

    def _process_start_trace(self, run: "Run") -> None:
        run_dict: Dict[str, Any] = run.dict()
        if not run.parent_run_id:
            # This is the first run, which maps to a chain
            chain_: "Chain" = self._chain.Chain(
                inputs=run_dict["inputs"],
                metadata=None,
                experiment_info=self._experiment_info.get(),
            )
            self._chains_map[run.id] = chain_
        else:
            span: "Span" = self._span.Span(
                inputs=run_dict["inputs"],
                category=_get_run_type(run),
                metadata=run_dict["extra"],
                name=run.name,
            )
            span.__api__start__(self._chains_map[run.parent_run_id])
            self._chains_map[run.id] = self._chains_map[run.parent_run_id]
            self._span_map[run.id] = span

    def _process_end_trace(self, run: "Run") -> None:
        run_dict: Dict[str, Any] = run.dict()
        if not run.parent_run_id:
            pass
            # Langchain will call _persist_run for us
        else:
            span = self._span_map[run.id]
            span.set_outputs(outputs=run_dict["outputs"])
            span.__api__end__()

    def flush(self) -> None:
        self._flush()

    def _on_llm_start(self, run: "Run") -> None:
        """Process the LLM Run upon start."""
        self._process_start_trace(run)

    def _on_llm_end(self, run: "Run") -> None:
        """Process the LLM Run."""
        self._process_end_trace(run)

    def _on_llm_error(self, run: "Run") -> None:
        """Process the LLM Run upon error."""
        self._process_end_trace(run)

    def _on_chain_start(self, run: "Run") -> None:
        """Process the Chain Run upon start."""
        self._process_start_trace(run)

    def _on_chain_end(self, run: "Run") -> None:
        """Process the Chain Run."""
        self._process_end_trace(run)

    def _on_chain_error(self, run: "Run") -> None:
        """Process the Chain Run upon error."""
        self._process_end_trace(run)

    def _on_tool_start(self, run: "Run") -> None:
        """Process the Tool Run upon start."""
        self._process_start_trace(run)

    def _on_tool_end(self, run: "Run") -> None:
        """Process the Tool Run."""
        self._process_end_trace(run)

    def _on_tool_error(self, run: "Run") -> None:
        """Process the Tool Run upon error."""
        self._process_end_trace(run)
