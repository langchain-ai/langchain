from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.plan_and_execute.executors.base import BaseExecutor
from langchain.agents.plan_and_execute.planners.base import BasePlanner
from langchain.agents.plan_and_execute.schema import Step, StepResponse
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class PlanAndExecute(Chain):
    planner: BasePlanner
    executer: BaseExecutor
    input_key: str = "input"
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        plan = self.planner.plan(
            inputs,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        if run_manager:
            run_manager.on_text(str(plan))
        previous_steps: List[Tuple[Step, StepResponse]] = []
        for step in plan.steps:
            _new_inputs = {"previous_steps": previous_steps, "current_step": step}
            new_inputs = {**_new_inputs, **inputs}
            response = self.executer.step(
                new_inputs,
                callbacks=run_manager.get_child() if run_manager else None,
            )
            if run_manager:
                run_manager.on_text(f"*****\n\nStep: {step.value}")
                run_manager.on_text(f"\n\nResponse: {response.response}")
            previous_steps.append((step, response))
        return {self.output_key: previous_steps[-1][1].response}
