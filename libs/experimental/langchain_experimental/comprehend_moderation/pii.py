import asyncio
from typing import Any, Dict, Optional

from langchain_experimental.comprehend_moderation.base_moderation_exceptions import (
    ModerationPiiError,
)


class ComprehendPII:
    """Class to handle Personally Identifiable Information (PII) moderation."""

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
            "moderation_type": "PII",
            "moderation_status": "LABELS_NOT_FOUND",
        }
        self.callback = callback
        self.unique_id = unique_id

    def validate(self, prompt_value: str, config: Any = None) -> str:
        redact = config.get("redact")
        return (
            self._detect_pii(prompt_value=prompt_value, config=config)
            if redact
            else self._contains_pii(prompt_value=prompt_value, config=config)
        )

    def _contains_pii(self, prompt_value: str, config: Any = None) -> str:
        """
        Checks for Personally Identifiable Information (PII) labels above a
        specified threshold. Uses Amazon Comprehend Contains PII Entities API. See -
        https://docs.aws.amazon.com/comprehend/latest/APIReference/API_ContainsPiiEntities.html
        Args:
            prompt_value (str): The input text to be checked for PII labels.
            config (Dict[str, Any]): Configuration for PII check and actions.

        Returns:
            str: the original prompt

        Note:
            - The provided client should be initialized with valid AWS credentials.
        """
        pii_identified = self.client.contains_pii_entities(
            Text=prompt_value, LanguageCode="en"
        )

        if self.callback and self.callback.pii_callback:
            self.moderation_beacon["moderation_input"] = prompt_value
            self.moderation_beacon["moderation_output"] = pii_identified

        threshold = config.get("threshold")
        pii_labels = config.get("labels")
        pii_found = False
        for entity in pii_identified["Labels"]:
            if (entity["Score"] >= threshold and entity["Name"] in pii_labels) or (
                entity["Score"] >= threshold and not pii_labels
            ):
                pii_found = True
                break

        if self.callback and self.callback.pii_callback:
            if pii_found:
                self.moderation_beacon["moderation_status"] = "LABELS_FOUND"
            asyncio.create_task(
                self.callback.on_after_pii(self.moderation_beacon, self.unique_id)
            )
        if pii_found:
            raise ModerationPiiError
        return prompt_value

    def _detect_pii(self, prompt_value: str, config: Optional[Dict[str, Any]]) -> str:
        """
        Detects and handles Personally Identifiable Information (PII) entities in the
        given prompt text using Amazon Comprehend's detect_pii_entities API. The
        function provides options to redact or stop processing based on the identified
        PII entities and a provided configuration. Uses Amazon Comprehend Detect PII
        Entities API.

        Args:
            prompt_value (str): The input text to be checked for PII entities.
            config (Dict[str, Any]): A configuration specifying how to handle
                                     PII entities.

        Returns:
            str: The processed prompt text with redacted PII entities or raised
                 exceptions.

        Raises:
            ValueError: If the prompt contains configured PII entities for
                        stopping processing.

        Note:
            - If PII is not found in the prompt, the original prompt is returned.
            - The client should be initialized with valid AWS credentials.
        """
        pii_identified = self.client.detect_pii_entities(
            Text=prompt_value, LanguageCode="en"
        )

        if self.callback and self.callback.pii_callback:
            self.moderation_beacon["moderation_input"] = prompt_value
            self.moderation_beacon["moderation_output"] = pii_identified

        if (pii_identified["Entities"]) == []:
            if self.callback and self.callback.pii_callback:
                asyncio.create_task(
                    self.callback.on_after_pii(self.moderation_beacon, self.unique_id)
                )
            return prompt_value

        pii_found = False
        if not config and pii_identified["Entities"]:
            for entity in pii_identified["Entities"]:
                if entity["Score"] >= 0.5:
                    pii_found = True
                    break

            if self.callback and self.callback.pii_callback:
                if pii_found:
                    self.moderation_beacon["moderation_status"] = "LABELS_FOUND"
                asyncio.create_task(
                    self.callback.on_after_pii(self.moderation_beacon, self.unique_id)
                )
            if pii_found:
                raise ModerationPiiError
        else:
            threshold = config.get("threshold")  # type: ignore
            pii_labels = config.get("labels")  # type: ignore
            mask_marker = config.get("mask_character")  # type: ignore
            pii_found = False

            for entity in pii_identified["Entities"]:
                if (
                    pii_labels
                    and entity["Type"] in pii_labels
                    and entity["Score"] >= threshold
                ) or (not pii_labels and entity["Score"] >= threshold):
                    pii_found = True
                    char_offset_begin = entity["BeginOffset"]
                    char_offset_end = entity["EndOffset"]

                    mask_length = char_offset_end - char_offset_begin + 1
                    masked_part = mask_marker * mask_length

                    prompt_value = (
                        prompt_value[:char_offset_begin]
                        + masked_part
                        + prompt_value[char_offset_end + 1 :]
                    )

            if self.callback and self.callback.pii_callback:
                if pii_found:
                    self.moderation_beacon["moderation_status"] = "LABELS_FOUND"
                asyncio.create_task(
                    self.callback.on_after_pii(self.moderation_beacon, self.unique_id)
                )

        return prompt_value
