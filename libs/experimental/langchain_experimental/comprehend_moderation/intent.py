import asyncio
import warnings
from typing import Any, Dict, Optional

from langchain_experimental.comprehend_moderation.base_moderation_exceptions import (
    ModerationIntentionError,
)


class ComprehendIntent:
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
            "moderation_type": "Intent",
            "moderation_status": "LABELS_NOT_FOUND",
        }
        self.callback = callback
        self.unique_id = unique_id

    def _get_arn(self) -> str:
        region_name = self.client.meta.region_name
        service = "comprehend"
        intent_endpoint = "document-classifier-endpoint/prompt-intent"
        return f"arn:aws:{service}:{region_name}:aws:{intent_endpoint}"

    def validate(
        self, prompt_value: str, config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Check and validate the intent of the given prompt text.

        Args:
            comprehend_client: Comprehend client for intent classification
            prompt_value (str): The input text to be checked for unintended intent
            config (Dict[str, Any]): Configuration settings for intent checks

        Raises:
            ValueError: If unintended intent is found in the prompt text based
                        on the specified threshold.

        Returns:
            str: The input prompt_value.

        Note:
            This function checks the intent of the provided prompt text using
            Comprehend's classify_document API and raises an error if unintended
            intent is detected with a score above the specified threshold.

        """
        from langchain_experimental.comprehend_moderation.base_moderation_enums import (
            BaseModerationActions,
        )

        threshold = config.get("threshold", 0.5) if config else 0.5
        action = (
            config.get("action", BaseModerationActions.STOP)
            if config
            else BaseModerationActions.STOP
        )
        intent_found = False

        if action == BaseModerationActions.ALLOW:
            warnings.warn(
                "You have allowed content with Harmful content."
                "Defaulting to STOP action..."
            )
            action = BaseModerationActions.STOP

        endpoint_arn = self._get_arn()
        response = self.client.classify_document(
            Text=prompt_value, EndpointArn=endpoint_arn
        )

        if self.callback and self.callback.intent_callback:
            self.moderation_beacon["moderation_input"] = prompt_value
            self.moderation_beacon["moderation_output"] = response

        for class_result in response["Classes"]:
            if (
                class_result["Score"] >= threshold
                and class_result["Name"] == "UNDESIRED_PROMPT"
            ):
                intent_found = True
                break

        if self.callback and self.callback.intent_callback:
            if intent_found:
                self.moderation_beacon["moderation_status"] = "LABELS_FOUND"
            asyncio.create_task(
                self.callback.on_after_intent(self.moderation_beacon, self.unique_id)
            )
        if intent_found:
            raise ModerationIntentionError
        return prompt_value
