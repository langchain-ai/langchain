import asyncio
from typing import Any, Optional

from langchain_experimental.comprehend_moderation.base_moderation_exceptions import (
    ModerationPromptSafetyError,
)


class ComprehendPromptSafety:
    """Class to handle prompt safety moderation."""

    def __init__(
        self,
        client: Any,
        callback: Optional[Any] = None,
        unique_id: Optional[str] = None,
        chain_id: Optional[str] = None,
    ) -> None:
        self.client = client
        self.moderation_beacon = {
            "moderation_chain_id": chain_id,
            "moderation_type": "PromptSafety",
            "moderation_status": "LABELS_NOT_FOUND",
        }
        self.callback = callback
        self.unique_id = unique_id

    def _get_arn(self) -> str:
        region_name = self.client.meta.region_name
        service = "comprehend"
        prompt_safety_endpoint = "document-classifier-endpoint/prompt-safety"
        return f"arn:aws:{service}:{region_name}:aws:{prompt_safety_endpoint}"

    def validate(self, prompt_value: str, config: Any = None) -> str:
        """
        Check and validate the safety of the given prompt text.

        Args:
            prompt_value (str): The input text to be checked for unsafe text.
            config (Dict[str, Any]): Configuration settings for prompt safety checks.

        Raises:
            ValueError: If unsafe prompt is found in the prompt text based
            on the specified threshold.

        Returns:
            str: The input prompt_value.

        Note:
            This function checks the safety of the provided prompt text using
            Comprehend's classify_document API and raises an error if unsafe
            text is detected with a score above the specified threshold.

        Example:
            comprehend_client = boto3.client('comprehend')
            prompt_text = "Please tell me your credit card information."
            config = {"threshold": 0.7}
            checked_prompt = check_prompt_safety(comprehend_client, prompt_text, config)
        """

        threshold = config.get("threshold")
        unsafe_prompt = False

        endpoint_arn = self._get_arn()
        response = self.client.classify_document(
            Text=prompt_value, EndpointArn=endpoint_arn
        )

        if self.callback and self.callback.prompt_safety_callback:
            self.moderation_beacon["moderation_input"] = prompt_value
            self.moderation_beacon["moderation_output"] = response

        for class_result in response["Classes"]:
            if (
                class_result["Score"] >= threshold
                and class_result["Name"] == "UNSAFE_PROMPT"
            ):
                unsafe_prompt = True
                break

        if self.callback and self.callback.intent_callback:
            if unsafe_prompt:
                self.moderation_beacon["moderation_status"] = "LABELS_FOUND"
            asyncio.create_task(
                self.callback.on_after_intent(self.moderation_beacon, self.unique_id)
            )
        if unsafe_prompt:
            raise ModerationPromptSafetyError
        return prompt_value
