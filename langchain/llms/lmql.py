"""Wrapper around LMQL"""
import logging
from typing import Dict, Optional, List, Any
import asyncio

from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from pydantic import Extra, root_validator

logger = logging.getLogger(__name__)

class LMQL(BaseLLM):
    """Wrapper around LMQL (Language Model Query Language)

    To use this, you need to have `lmql` Python library installed. 
    Check out: https://github.com/eth-sri/lmql
    
    Example:
        .. code-block:: python

            from langchain.llms import LMQL
            llm = LMQL()
    """

    class Config:
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "lmql"
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            import lmql
        
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import lmql python package."
                "Please install with `pip install lmql`."
            )
        return values
    
    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        return asyncio.run(self.agenerate(prompts=prompts, stop=stop))
    
    async def _agenerate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        for prompt in prompts:
            result = await self._call_lmql(prompt)
            generation = []
            for i in result:
                cur = Generation(text=i.prompt, generation_info=i.variables)
                generation.append(cur)
            generations.append(generation)
        return LLMResult(generations=generations)

    
    async def _call_lmql(self, query_str: str) -> List[Dict[str, Any]]:
        import lmql
        query_func = lmql.query(query_str)
        # LMQL can return a list of LMQLResult or just LMQLResult
        result = await query_func()
        if type(result) != list:
            result = [result]
        return result

    

