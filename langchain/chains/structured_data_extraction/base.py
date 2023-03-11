from __future__ import annotations
from typing import Dict, List, Any

from langchain.chains.base import Chain
from pydantic import root_validator



class StructuredDataExtractionChain(Chain):


    kor_extractor: Any
    kor_schema: Any
    input_key: str = "text"
    output_key: str = "info"
    @property
    def _chain_type(self) -> str:
        raise NotImplementedError

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import kor
        except ImportError:
            raise ValueError(
                "Could not import kor python package. "
                "Please it install it with `pip install kor`."
            )
        return values

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        result = self.kor_extractor(inputs[self.input_key], self.kor_schema)
        return {self.output_key: result}

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        pass