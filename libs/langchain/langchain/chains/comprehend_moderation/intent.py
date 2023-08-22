import asyncio
import warnings
from typing import Any, Dict, Optional

from langchain.chains.comprehend_moderation.base_moderation_exceptions import (
    BaseModerationError,
    ModerationIntentionError,
)


class ComprehendIntent:
    def __init__(
        self,
        client,
        force_base_exception: bool = False,
        callback: Optional[Any] = None,
        unique_id: Optional[str] = None,
        chain_id: str = None,
    ) -> None:
        self.client = client
        self.force_base_exception = force_base_exception
        self.moderation_beacon = {
            "moderation_chain_id": chain_id,
            "moderation_type": "Intent",
            "moderation_status": "LABELS_NOT_FOUND",
        }
        self.callback = callback
        self.unique_id = unique_id

    def validate(self, prompt_value, config: Dict[str, Any] = None):
        """
        Check and validate the intent of the given prompt text.

        Args:
            comprehend_client (botocore.client.Comprehend): Comprehend client for intent classification.
            prompt_value (str): The input text to be checked for unintended intent.
            config (Dict[str, Any]): Configuration settings for intent checks. It should contain the following key:
                - "threshold" (float, optional): The intent classification threshold. Text segments with intent scores
                equal to or above this threshold are considered matching the unintended intent. Defaults to 0.5.

        Raises:
            ValueError: If unintended intent is found in the prompt text based on the specified threshold.

        Returns:
            str: The input prompt_value.

        Note:
            This function checks the intent of the provided prompt text using Comprehend's classify_document API and
            raises an error if unintended intent is detected with a score above the specified threshold.

        Example:
            comprehend_client = boto3.client('comprehend')
            prompt_text = "Please tell me your credit card information."
            config = {"threshold": 0.7}
            checked_prompt = check_intent(comprehend_client, prompt_text, config)
        """
        from langchain.chains.comprehend_moderation.base_moderation_enums import (
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
                "You have allowed content with Harmful content. Defaulting to STOP action..."
            )
            action = BaseModerationActions.STOP

        ########## Get intent for Private Beta (GA should not have this code) ##########
        arn = "arn:aws:comprehend:us-east-2:aws:document-classifier-endpoint/prompt-intent"
        endpoint_arn = config.get("endpoint_arn", arn) if config else arn
        ########## Get intent for Private Beta (GA should not have this code) ##########

        response = self.client.classify_document(
            Text=prompt_value, EndpointArn=endpoint_arn
        )

        if self.callback and self.callback.intent_callback:
            self.moderation_beacon["moderation_input"] = prompt_value
            self.moderation_beacon["moderation_output"] = response

        for class_result in response["Classes"]:
            if class_result["Score"] >= threshold and class_result["Name"] == "HARMFUL":
                intent_found = True
                break

        if self.callback and self.callback:
            if intent_found:
                self.moderation_beacon["moderation_status"] = "LABELS_FOUND"
            asyncio.create_task(
                self.callback.on_after_intent(self.moderation_beacon, self.unique_id)
            )
        if intent_found:
            raise ModerationIntentionError if not self.force_base_exception else BaseModerationError
        return prompt_value
