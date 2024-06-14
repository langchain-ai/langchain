import logging
import re
import string
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.pydantic_v1 import root_validator

from langchain_community.llms.vertexai import _VertexAICommon
from langchain_community.utilities.vertexai import raise_vertex_import_error

logger = logging.getLogger(__name__)

_MAX_TOKENS_PER_BATCH = 20000
_MAX_BATCH_SIZE = 250
_MIN_BATCH_SIZE = 5


@deprecated(
    since="0.0.12",
    removal="0.3.0",
    alternative_import="langchain_google_vertexai.VertexAIEmbeddings",
)
class VertexAIEmbeddings(_VertexAICommon, Embeddings):
    """Google Cloud VertexAI embedding models."""

    # Instance context
    instance: Dict[str, Any] = {}  #: :meta private:
    show_progress_bar: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        cls._try_init_vertexai(values)
        if values["model_name"] == "textembedding-gecko-default":
            logger.warning(
                "Model_name will become a required arg for VertexAIEmbeddings "
                "starting from Feb-01-2024. Currently the default is set to "
                "textembedding-gecko@001"
            )
            values["model_name"] = "textembedding-gecko@001"
        try:
            from vertexai.language_models import TextEmbeddingModel
        except ImportError:
            raise_vertex_import_error()
        values["client"] = TextEmbeddingModel.from_pretrained(values["model_name"])
        return values

    def __init__(
        self,
        # the default value would be removed after Feb-01-2024
        model_name: str = "textembedding-gecko-default",
        project: Optional[str] = None,
        location: str = "us-central1",
        request_parallelism: int = 5,
        max_retries: int = 6,
        credentials: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize the sentence_transformer."""
        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            request_parallelism=request_parallelism,
            max_retries=max_retries,
            model_name=model_name,
            **kwargs,
        )
        self.instance["max_batch_size"] = kwargs.get("max_batch_size", _MAX_BATCH_SIZE)
        self.instance["batch_size"] = self.instance["max_batch_size"]
        self.instance["min_batch_size"] = kwargs.get("min_batch_size", _MIN_BATCH_SIZE)
        self.instance["min_good_batch_size"] = self.instance["min_batch_size"]
        self.instance["lock"] = threading.Lock()
        self.instance["batch_size_validated"] = False
        self.instance["task_executor"] = ThreadPoolExecutor(
            max_workers=request_parallelism
        )
        self.instance[
            "embeddings_task_type_supported"
        ] = not self.client._endpoint_name.endswith("/textembedding-gecko@001")

    @staticmethod
    def _split_by_punctuation(text: str) -> List[str]:
        """Splits a string by punctuation and whitespace characters."""
        split_by = string.punctuation + "\t\n "
        pattern = f"([{split_by}])"
        # Using re.split to split the text based on the pattern
        return [segment for segment in re.split(pattern, text) if segment]

    @staticmethod
    def _prepare_batches(texts: List[str], batch_size: int) -> List[List[str]]:
        """Splits texts in batches based on current maximum batch size
        and maximum tokens per request.
        """
        text_index = 0
        texts_len = len(texts)
        batch_token_len = 0
        batches: List[List[str]] = []
        current_batch: List[str] = []
        if texts_len == 0:
            return []
        while text_index < texts_len:
            current_text = texts[text_index]
            # Number of tokens per a text is conservatively estimated
            # as 2 times number of words, punctuation and whitespace characters.
            # Using `count_tokens` API will make batching too expensive.
            # Utilizing a tokenizer, would add a dependency that would not
            # necessarily be reused by the application using this class.
            current_text_token_cnt = (
                len(VertexAIEmbeddings._split_by_punctuation(current_text)) * 2
            )
            end_of_batch = False
            if current_text_token_cnt > _MAX_TOKENS_PER_BATCH:
                # Current text is too big even for a single batch.
                # Such request will fail, but we still make a batch
                # so that the app can get the error from the API.
                if len(current_batch) > 0:
                    # Adding current batch if not empty.
                    batches.append(current_batch)
                current_batch = [current_text]
                text_index += 1
                end_of_batch = True
            elif (
                batch_token_len + current_text_token_cnt > _MAX_TOKENS_PER_BATCH
                or len(current_batch) == batch_size
            ):
                end_of_batch = True
            else:
                if text_index == texts_len - 1:
                    # Last element - even though the batch may be not big,
                    # we still need to make it.
                    end_of_batch = True
                batch_token_len += current_text_token_cnt
                current_batch.append(current_text)
                text_index += 1
            if end_of_batch:
                batches.append(current_batch)
                current_batch = []
                batch_token_len = 0
        return batches

    def _get_embeddings_with_retry(
        self, texts: List[str], embeddings_type: Optional[str] = None
    ) -> List[List[float]]:
        """Makes a Vertex AI model request with retry logic."""
        from google.api_core.exceptions import (
            Aborted,
            DeadlineExceeded,
            ResourceExhausted,
            ServiceUnavailable,
        )

        errors = [
            ResourceExhausted,
            ServiceUnavailable,
            Aborted,
            DeadlineExceeded,
        ]
        retry_decorator = create_base_retry_decorator(
            error_types=errors, max_retries=self.max_retries
        )

        @retry_decorator
        def _completion_with_retry(texts_to_process: List[str]) -> Any:
            if embeddings_type and self.instance["embeddings_task_type_supported"]:
                from vertexai.language_models import TextEmbeddingInput

                requests = [
                    TextEmbeddingInput(text=t, task_type=embeddings_type)
                    for t in texts_to_process
                ]
            else:
                requests = texts_to_process
            embeddings = self.client.get_embeddings(requests)
            return [embs.values for embs in embeddings]

        return _completion_with_retry(texts)

    def _prepare_and_validate_batches(
        self, texts: List[str], embeddings_type: Optional[str] = None
    ) -> Tuple[List[List[float]], List[List[str]]]:
        """Prepares text batches with one-time validation of batch size.
        Batch size varies between GCP regions and individual project quotas.
        # Returns embeddings of the first text batch that went through,
        # and text batches for the rest of the texts.
        """
        from google.api_core.exceptions import InvalidArgument

        batches = VertexAIEmbeddings._prepare_batches(
            texts, self.instance["batch_size"]
        )
        # If batch size if less or equal to one that went through before,
        # then keep batches as they are.
        if len(batches[0]) <= self.instance["min_good_batch_size"]:
            return [], batches
        with self.instance["lock"]:
            # If largest possible batch size was validated
            # while waiting for the lock, then check for rebuilding
            # our batches, and return.
            if self.instance["batch_size_validated"]:
                if len(batches[0]) <= self.instance["batch_size"]:
                    return [], batches
                else:
                    return [], VertexAIEmbeddings._prepare_batches(
                        texts, self.instance["batch_size"]
                    )
            # Figure out largest possible batch size by trying to push
            # batches and lowering their size in half after every failure.
            first_batch = batches[0]
            first_result = []
            had_failure = False
            while True:
                try:
                    first_result = self._get_embeddings_with_retry(
                        first_batch, embeddings_type
                    )
                    break
                except InvalidArgument:
                    had_failure = True
                    first_batch_len = len(first_batch)
                    if first_batch_len == self.instance["min_batch_size"]:
                        raise
                    first_batch_len = max(
                        self.instance["min_batch_size"], int(first_batch_len / 2)
                    )
                    first_batch = first_batch[:first_batch_len]
            first_batch_len = len(first_batch)
            self.instance["min_good_batch_size"] = max(
                self.instance["min_good_batch_size"], first_batch_len
            )
            # If had a failure and recovered
            # or went through with the max size, then it's a legit batch size.
            if had_failure or first_batch_len == self.instance["max_batch_size"]:
                self.instance["batch_size"] = first_batch_len
                self.instance["batch_size_validated"] = True
                # If batch size was updated,
                # rebuild batches with the new batch size
                # (texts that went through are excluded here).
                if first_batch_len != self.instance["max_batch_size"]:
                    batches = VertexAIEmbeddings._prepare_batches(
                        texts[first_batch_len:], self.instance["batch_size"]
                    )
            else:
                # Still figuring out max batch size.
                batches = batches[1:]
        # Returning embeddings of the first text batch that went through,
        # and text batches for the rest of texts.
        return first_result, batches

    def embed(
        self,
        texts: List[str],
        batch_size: int = 0,
        embeddings_task_type: Optional[
            Literal[
                "RETRIEVAL_QUERY",
                "RETRIEVAL_DOCUMENT",
                "SEMANTIC_SIMILARITY",
                "CLASSIFICATION",
                "CLUSTERING",
            ]
        ] = None,
    ) -> List[List[float]]:
        """Embed a list of strings.

        Args:
            texts: List[str] The list of strings to embed.
            batch_size: [int] The batch size of embeddings to send to the model.
                If zero, then the largest batch size will be detected dynamically
                at the first request, starting from 250, down to 5.
            embeddings_task_type: [str] optional embeddings task type,
                one of the following
                    RETRIEVAL_QUERY	- Text is a query
                                      in a search/retrieval setting.
                    RETRIEVAL_DOCUMENT - Text is a document
                                         in a search/retrieval setting.
                    SEMANTIC_SIMILARITY - Embeddings will be used
                                          for Semantic Textual Similarity (STS).
                    CLASSIFICATION - Embeddings will be used for classification.
                    CLUSTERING - Embeddings will be used for clustering.

        Returns:
            List of embeddings, one for each text.
        """
        if len(texts) == 0:
            return []
        embeddings: List[List[float]] = []
        first_batch_result: List[List[float]] = []
        if batch_size > 0:
            # Fixed batch size.
            batches = VertexAIEmbeddings._prepare_batches(texts, batch_size)
        else:
            # Dynamic batch size, starting from 250 at the first call.
            first_batch_result, batches = self._prepare_and_validate_batches(
                texts, embeddings_task_type
            )
        # First batch result may have some embeddings already.
        # In such case, batches have texts that were not processed yet.
        embeddings.extend(first_batch_result)
        tasks = []
        if self.show_progress_bar:
            try:
                from tqdm import tqdm

                iter_ = tqdm(batches, desc="VertexAIEmbeddings")
            except ImportError:
                logger.warning(
                    "Unable to show progress bar because tqdm could not be imported. "
                    "Please install with `pip install tqdm`."
                )
                iter_ = batches
        else:
            iter_ = batches
        for batch in iter_:
            tasks.append(
                self.instance["task_executor"].submit(
                    self._get_embeddings_with_retry,
                    texts=batch,
                    embeddings_type=embeddings_task_type,
                )
            )
        if len(tasks) > 0:
            wait(tasks)
        for t in tasks:
            embeddings.extend(t.result())
        return embeddings

    def embed_documents(
        self, texts: List[str], batch_size: int = 0
    ) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List[str] The list of texts to embed.
            batch_size: [int] The batch size of embeddings to send to the model.
                If zero, then the largest batch size will be detected dynamically
                at the first request, starting from 250, down to 5.

        Returns:
            List of embeddings, one for each text.
        """
        return self.embed(texts, batch_size, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embeddings = self.embed([text], 1, "RETRIEVAL_QUERY")
        return embeddings[0]
