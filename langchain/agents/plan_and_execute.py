from typing import Dict, Any, Optional, List

from pydantic import BaseModel
from abc import abstractmethod

from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun, Callbacks
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.schema import BaseOutputParser




class Plan(BaseModel):
    steps: List[str]


class Step(BaseModel):
    response: str


class PlanOutputParser(BaseOutputParser):

    def parse(self, text: str) -> Plan:
        """Parse into a plan."""


class BasePlanner(BaseModel):
    @abstractmethod
    def plan(
        self,
        inputs: dict,
        callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Plan:
        """Given input, decided what to do."""

    @abstractmethod
    async def aplan(
        self,
        inputs: dict,
        callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Plan:
        """Given input, decided what to do. """


class BaseStepper(BaseModel):
    @abstractmethod
    def step(
        self,
        inputs: dict,
        callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Step:
        """Take step."""

    @abstractmethod
    async def astep(
        self,
        inputs: dict,
        callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Step:
        """Take step."""

class LLMPlanner(BasePlanner):
    llm_chain: LLMChain
    output_parser: PlanOutputParser

    def plan(
        self,
        inputs: dict,
        callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Plan:
        """Given input, decided what to do."""
        llm_response = self.llm_chain.run(**inputs, callbacks=callbacks)
        return self.output_parser.parse(llm_response)

    async def aplan(
        self,
        inputs: dict,
        callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Plan:
        """Given input, decided what to do. """
        llm_response = await self.llm_chain.arun(**inputs, callbacks=callbacks)
        return self.output_parser.parse(llm_response)

class LLMChainStepper(BaseStepper):
    llm_chain: LLMChain

    def step(
        self,
        inputs: dict,
        callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Step:
        """Take step."""
        response = self.llm_chain.run(**inputs, callbacks=callbacks)
        return Step(response=response)

    async def astep(
        self,
        inputs: dict,
        callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Step:
        """Take step."""
        response = await self.llm_chain.arun(**inputs, callbacks=callbacks)
        return Step(response=response)


class PlanAndExecute(Chain):
    planner: BasePlanner
    executer: BaseStepper

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        plan = self.planner.plan(inputs, callbacks=run_manager.get_child() if run_manager else None,)
        print(plan)
        previous_steps = []
        for step in plan.steps:
            _new_inputs = {"previous_steps": previous_steps, "current_step": step}
            new_inputs = {**_new_inputs, **inputs}
            response = self.executer.step(new_inputs, callbacks=run_manager.get_child() if run_manager else None,)
            print(response)
            previous_steps.append((step, response))
        return {"answer": previous_steps[-1][1].response}
