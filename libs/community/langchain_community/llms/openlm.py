from typing import Any, Dict

from langchain_core.utils import pre_init

from langchain_community.llms.openai import BaseOpenAI


class OpenLM(BaseOpenAI):
    """OpenLM models."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            import openlm

            values["client"] = openlm.Completion
        except ImportError:
            raise ImportError(
                "Could not import openlm python package. "
                "Please install it with `pip install openlm`."
            )
        if values["streaming"]:
            raise ValueError("Streaming not supported with openlm")
        return values
