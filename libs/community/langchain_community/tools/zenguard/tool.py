import os
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError, validator
from langchain_core.tools import BaseTool


class Detector(str, Enum):
    ALLOWED_TOPICS = "allowed_subjects"
    BANNED_TOPICS = "banned_subjects"
    PROMPT_INJECTION = "prompt_injection"
    KEYWORDS = "keywords"
    PII = "pii"
    SECRETS = "secrets"
    TOXICITY = "toxicity"


class DetectorAPI(str, Enum):
    ALLOWED_TOPICS = "v1/detect/topics/allowed"
    BANNED_TOPICS = "v1/detect/topics/banned"
    PROMPT_INJECTION = "v1/detect/prompt_injection"
    KEYWORDS = "v1/detect/keywords"
    PII = "v1/detect/pii"
    SECRETS = "v1/detect/secrets"
    TOXICITY = "v1/detect/toxicity"


class ZenGuardInput(BaseModel):
    prompts: List[str] = Field(
        ...,
        min_items=1,
        min_length=1,
        description="Prompt to check",
    )
    detectors: List[Detector] = Field(
        ...,
        min_items=1,
        description="List of detectors by which you want to check the prompt",
    )
    in_parallel: bool = Field(
        default=True,
        description="Run prompt detection by the detector in parallel or sequentially",
    )


class ZenGuardTool(BaseTool):
    name: str = "ZenGuard"
    description: str = (
        "ZenGuard AI integration package. ZenGuard AI - the fastest GenAI guardrails."
    )
    args_schema = ZenGuardInput
    return_direct: bool = True

    zenguard_api_key: Optional[str] = Field(default=None)

    _ZENGUARD_API_URL_ROOT: str = "https://api.zenguard.ai/"
    _ZENGUARD_API_KEY_ENV_NAME: str = "ZENGUARD_API_KEY"

    @validator("zenguard_api_key", pre=True, always=True, check_fields=False)
    def set_api_key(cls, v: str) -> str:
        if v is None:
            v = os.getenv(cls._ZENGUARD_API_KEY_ENV_NAME)
        if v is None:
            raise ValidationError(
                "The zenguard_api_key tool option must be set either "
                "by passing zenguard_api_key to the tool or by setting "
                f"the f{cls._ZENGUARD_API_KEY_ENV_NAME} environment variable"
            )
        return v

    def _run(
        self,
        prompts: List[str],
        detectors: List[Detector],
        in_parallel: bool = True,
    ) -> Dict[str, Any]:
        try:
            postfix = None
            json: Optional[Dict[str, Any]] = None
            if len(detectors) == 1:
                postfix = self._convert_detector_to_api(detectors[0])
                json = {"messages": prompts}
            else:
                postfix = "v1/detect"
                json = {
                    "messages": prompts,
                    "in_parallel": in_parallel,
                    "detectors": detectors,
                }
            response = requests.post(
                self._ZENGUARD_API_URL_ROOT + postfix,
                json=json,
                headers={"x-api-key": self.zenguard_api_key},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except (requests.HTTPError, requests.Timeout) as e:
            return {"error": str(e)}

    def _convert_detector_to_api(self, detector: Detector) -> str:
        return DetectorAPI[detector.name].value
