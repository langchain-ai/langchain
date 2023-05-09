from typing import Dict, Any, Optional, List

from pydantic import BaseModel

from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain.chains.base import Chain


class Plan(BaseModel):
    steps: List[str]



class PlanAndExecute(Chain):
    plan_chain: Chain
    execute_chain: Chain

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        plan_response = self.plan_chain(inputs, callbacks=run_manager.get_child() if run_manager else None,)
        plan = Plan.from_orm(plan_response)
        previous_steps = []
        for step in plan.steps:
            _new_inputs = {"previous_steps": previous_steps, "current_step": step}
            new_inputs = {**_new_inputs, **inputs}
            response = self.execute_chain(new_inputs, callbacks=run_manager.get_child() if run_manager else None,)
            previous_steps.append((step, response))
        return {"answer": previous_steps[-1][1]}
