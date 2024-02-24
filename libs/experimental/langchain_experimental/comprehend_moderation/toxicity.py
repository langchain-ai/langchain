import asyncio
import importlib
from typing import Any, List, Optional

from langchain_experimental.comprehend_moderation.base_moderation_exceptions import (
    ModerationToxicityError,
)


class ComprehendToxicity:
    """Class to handle toxicity moderation."""

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
            "moderation_type": "Toxicity",
            "moderation_status": "LABELS_NOT_FOUND",
        }
        self.callback = callback
        self.unique_id = unique_id

    def _toxicity_init_validate(self, max_size: int) -> Any:
        """
        Validate and initialize toxicity processing configuration.

        Args:
            max_size (int): Maximum sentence size defined in the
            configuration object.

        Raises:
            Exception: If the maximum sentence size exceeds the 5KB limit.

        Note:
            This function ensures that the NLTK punkt tokenizer is downloaded
            if not already present.

        Returns:
            None
        """
        if max_size > 1024 * 5:
            raise Exception("The sentence length should not exceed 5KB.")
        try:
            nltk = importlib.import_module("nltk")
            nltk.data.find("tokenizers/punkt")
            return nltk
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import nltk python package. "
                "Please install it with `pip install nltk`."
            )
        except LookupError:
            nltk.download("punkt")

    def _split_paragraph(
        self, prompt_value: str, max_size: int = 1024 * 4
    ) -> List[List[str]]:
        """
        Split a paragraph into chunks of sentences, respecting the maximum size limit.

        Args:
            paragraph (str): The input paragraph to be split into chunks.
            max_size (int, optional): The maximum size limit in bytes for
            each chunk. Defaults to 1024.

        Returns:
            List[List[str]]: A list of chunks, where each chunk is a list
            of sentences.

        Note:
            This function validates the maximum sentence size based on service
            limits using the 'toxicity_init_validate' function. It uses the NLTK
            sentence tokenizer to split the paragraph into sentences.

        Example:
            paragraph = "This is a sample paragraph. It
            contains multiple sentences. ..."
            chunks = split_paragraph(paragraph, max_size=2048)
        """

        # validate max. sentence size based on Service limits
        nltk = self._toxicity_init_validate(max_size)
        sentences = nltk.sent_tokenize(prompt_value)
        chunks = list()  # type: ignore
        current_chunk = list()  # type: ignore
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.encode("utf-8"))
            # If adding a new sentence exceeds max_size
            # or current_chunk has 10 sentences, start a new chunk
            if (current_size + sentence_size > max_size) or (len(current_chunk) >= 10):
                if current_chunk:  # Avoid appending empty chunks
                    chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add any remaining sentences
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def validate(self, prompt_value: str, config: Any = None) -> str:
        """
        Check the toxicity of a given text prompt using AWS
        Comprehend service and apply actions based on configuration.
        Args:
            prompt_value (str): The text content to be checked for toxicity.
            config (Dict[str, Any]): Configuration for toxicity checks and actions.

        Returns:
            str: The original prompt_value if allowed or no toxicity found.

        Raises:
            ValueError: If the prompt contains toxic labels and cannot be
            processed based on the configuration.
        """

        chunks = self._split_paragraph(prompt_value=prompt_value)
        for sentence_list in chunks:
            segments = [{"Text": sentence} for sentence in sentence_list]
            response = self.client.detect_toxic_content(
                TextSegments=segments, LanguageCode="en"
            )
            if self.callback and self.callback.toxicity_callback:
                self.moderation_beacon["moderation_input"] = segments  # type: ignore
                self.moderation_beacon["moderation_output"] = response
            toxicity_found = False
            threshold = config.get("threshold")
            toxicity_labels = config.get("labels")

            if not toxicity_labels:
                for item in response["ResultList"]:
                    for label in item["Labels"]:
                        if label["Score"] >= threshold:
                            toxicity_found = True
                            break
            else:
                for item in response["ResultList"]:
                    for label in item["Labels"]:
                        if (
                            label["Name"] in toxicity_labels
                            and label["Score"] >= threshold
                        ):
                            toxicity_found = True
                            break

            if self.callback and self.callback.toxicity_callback:
                if toxicity_found:
                    self.moderation_beacon["moderation_status"] = "LABELS_FOUND"
                asyncio.create_task(
                    self.callback.on_after_toxicity(
                        self.moderation_beacon, self.unique_id
                    )
                )
            if toxicity_found:
                raise ModerationToxicityError
        return prompt_value
