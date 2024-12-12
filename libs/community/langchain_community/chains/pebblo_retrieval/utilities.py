import json
import logging
import os
import platform
import threading
import time
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientTimeout
from langchain_core.documents import Document
from langchain_core.env import get_runtime_environment
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel
from requests import Response, request
from requests.exceptions import RequestException

from langchain_community.chains.pebblo_retrieval.models import (
    App,
    AuthContext,
    Context,
    Framework,
    IdentityPolicy,
    Prompt,
    Qa,
    Runtime,
    SemanticContext,
)

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.1"

_DEFAULT_CLASSIFIER_URL = "http://localhost:8000"
_DEFAULT_PEBBLO_CLOUD_URL = "https://api.daxa.ai"
_POLICY_REFRESH_INTERVAL_SEC = 300  # 5 minutes


class Routes(str, Enum):
    """Routes available for the Pebblo API as enumerator."""

    retrieval_app_discover = "/v1/app/discover"
    prompt = "/v1/prompt"
    prompt_governance = "/v1/prompt/governance"
    policy = "/v1/app/policy"


class PolicyType(str, Enum):
    """Policy type enumerator."""

    IDENTITY = "identity"
    APPLICATION = "application"


def get_runtime() -> Tuple[Framework, Runtime]:
    """Fetch the current Framework and Runtime details.

    Returns:
        Tuple[Framework, Runtime]: Framework and Runtime for the current app instance.
    """
    runtime_env = get_runtime_environment()
    framework = Framework(
        name="langchain", version=runtime_env.get("library_version", None)
    )
    uname = platform.uname()
    runtime = Runtime(
        host=uname.node,
        path=os.environ["PWD"],
        platform=runtime_env.get("platform", "unknown"),
        os=uname.system,
        os_version=uname.version,
        ip=get_ip(),
        language=runtime_env.get("runtime", "unknown"),
        language_version=runtime_env.get("runtime_version", "unknown"),
    )

    if "Darwin" in runtime.os:
        runtime.type = "desktop"
        runtime.runtime = "Mac OSX"

    logger.debug(f"framework {framework}")
    logger.debug(f"runtime {runtime}")
    return framework, runtime


def get_ip() -> str:
    """Fetch local runtime ip address.

    Returns:
        str: IP address
    """
    import socket  # lazy imports

    host = socket.gethostname()
    try:
        public_ip = socket.gethostbyname(host)
    except Exception:
        public_ip = socket.gethostbyname("localhost")
    return public_ip


class PebbloRetrievalAPIWrapper(BaseModel):
    """Wrapper for Pebblo Retrieval API."""

    api_key: Optional[str]  # Use SecretStr
    """API key for Pebblo Cloud"""
    classifier_location: str = "local"
    """Location of the classifier, local or cloud. Defaults to 'local'"""
    classifier_url: Optional[str]
    """URL of the Pebblo Classifier"""
    cloud_url: Optional[str]
    """URL of the Pebblo Cloud"""
    app_name: str
    """Name of the app"""
    policy_cache: Optional[IdentityPolicy] = None
    """Local cache for the policy"""

    def __init__(self, **kwargs: Any):
        """Validate that api key in environment."""
        kwargs["api_key"] = get_from_dict_or_env(
            kwargs, "api_key", "PEBBLO_API_KEY", ""
        )
        kwargs["classifier_url"] = get_from_dict_or_env(
            kwargs, "classifier_url", "PEBBLO_CLASSIFIER_URL", _DEFAULT_CLASSIFIER_URL
        )
        kwargs["cloud_url"] = get_from_dict_or_env(
            kwargs, "cloud_url", "PEBBLO_CLOUD_URL", _DEFAULT_PEBBLO_CLOUD_URL
        )
        super().__init__(**kwargs)
        if self.api_key:
            # Start a thread to fetch policy from the Pebblo cloud
            self._start_policy_refresh_thread()

    def send_app_discover(self, app: App) -> None:
        """
        Send app discovery request to Pebblo server & cloud.

        Args:
            app (App): App instance to be discovered.
        """
        pebblo_resp = None
        payload = app.model_dump(exclude_unset=True)

        if self.classifier_location == "local":
            # Send app details to local classifier
            headers = self._make_headers()
            app_discover_url = (
                f"{self.classifier_url}{Routes.retrieval_app_discover.value}"
            )
            pebblo_resp = self.make_request("POST", app_discover_url, headers, payload)

        if self.api_key:
            # Send app details to Pebblo cloud if api_key is present
            headers = self._make_headers(cloud_request=True)
            if pebblo_resp:
                pebblo_server_version = json.loads(pebblo_resp.text).get(
                    "pebblo_server_version"
                )
                payload.update({"pebblo_server_version": pebblo_server_version})

            payload.update({"pebblo_client_version": PLUGIN_VERSION})
            pebblo_cloud_url = f"{self.cloud_url}{Routes.retrieval_app_discover.value}"
            _ = self.make_request("POST", pebblo_cloud_url, headers, payload)

    def send_prompt(
        self,
        app_name: str,
        retriever: VectorStoreRetriever,
        question: str,
        answer: str,
        auth_context: Optional[AuthContext],
        docs: List[Document],
        prompt_entities: Dict[str, Any],
        prompt_time: str,
        prompt_gov_enabled: bool = False,
    ) -> None:
        """
        Send prompt to Pebblo server for classification.
        Then send prompt to Daxa cloud(If api_key is present).

        Args:
            app_name (str): Name of the app.
            retriever (VectorStoreRetriever): Retriever instance.
            question (str): Question asked in the prompt.
            answer (str): Answer generated by the model.
            auth_context (Optional[AuthContext]): Authentication context.
            docs (List[Document]): List of documents retrieved.
            prompt_entities (Dict[str, Any]): Entities present in the prompt.
            prompt_time (str): Time when the prompt was generated.
            prompt_gov_enabled (bool): Whether prompt governance is enabled.
        """
        pebblo_resp = None
        payload = self.build_prompt_qa_payload(
            app_name,
            retriever,
            question,
            answer,
            auth_context,
            docs,
            prompt_entities,
            prompt_time,
            prompt_gov_enabled,
        )

        if self.classifier_location == "local":
            # Send prompt to local classifier
            headers = self._make_headers()
            prompt_url = f"{self.classifier_url}{Routes.prompt.value}"
            pebblo_resp = self.make_request("POST", prompt_url, headers, payload)

        if self.api_key:
            # Send prompt to Pebblo cloud if api_key is present
            if self.classifier_location == "local":
                # If classifier location is local, then response, context and prompt
                # should be fetched from pebblo_resp and replaced in payload.
                pebblo_resp = pebblo_resp.json() if pebblo_resp else None
                self.update_cloud_payload(payload, pebblo_resp)

            headers = self._make_headers(cloud_request=True)
            pebblo_cloud_prompt_url = f"{self.cloud_url}{Routes.prompt.value}"
            _ = self.make_request("POST", pebblo_cloud_prompt_url, headers, payload)
        elif self.classifier_location == "pebblo-cloud":
            logger.warning("API key is missing for sending prompt to Pebblo cloud.")
            raise NameError("API key is missing for sending prompt to Pebblo cloud.")

    async def asend_prompt(
        self,
        app_name: str,
        retriever: VectorStoreRetriever,
        question: str,
        answer: str,
        auth_context: Optional[AuthContext],
        docs: List[Document],
        prompt_entities: Dict[str, Any],
        prompt_time: str,
        prompt_gov_enabled: bool = False,
    ) -> None:
        """
        Send prompt to Pebblo server for classification.
        Then send prompt to Daxa cloud(If api_key is present).

        Args:
            app_name (str): Name of the app.
            retriever (VectorStoreRetriever): Retriever instance.
            question (str): Question asked in the prompt.
            answer (str): Answer generated by the model.
            auth_context (Optional[AuthContext]): Authentication context.
            docs (List[Document]): List of documents retrieved.
            prompt_entities (Dict[str, Any]): Entities present in the prompt.
            prompt_time (str): Time when the prompt was generated.
            prompt_gov_enabled (bool): Whether prompt governance is enabled.
        """
        pebblo_resp = None
        payload = self.build_prompt_qa_payload(
            app_name,
            retriever,
            question,
            answer,
            auth_context,
            docs,
            prompt_entities,
            prompt_time,
            prompt_gov_enabled,
        )

        if self.classifier_location == "local":
            # Send prompt to local classifier
            headers = self._make_headers()
            prompt_url = f"{self.classifier_url}{Routes.prompt.value}"
            pebblo_resp = await self.amake_request("POST", prompt_url, headers, payload)

        if self.api_key:
            # Send prompt to Pebblo cloud if api_key is present
            if self.classifier_location == "local":
                # If classifier location is local, then response, context and prompt
                # should be fetched from pebblo_resp and replaced in payload.
                self.update_cloud_payload(payload, pebblo_resp)

            headers = self._make_headers(cloud_request=True)
            pebblo_cloud_prompt_url = f"{self.cloud_url}{Routes.prompt.value}"
            _ = await self.amake_request(
                "POST", pebblo_cloud_prompt_url, headers, payload
            )
        elif self.classifier_location == "pebblo-cloud":
            logger.warning("API key is missing for sending prompt to Pebblo cloud.")
            raise NameError("API key is missing for sending prompt to Pebblo cloud.")

    def check_prompt_validity(self, question: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check the validity of the given prompt using a remote classification service.

        This method sends a prompt to a remote classifier service and return entities
        present in prompt or not.

        Args:
            question (str): The prompt question to be validated.

        Returns:
            bool: True if the prompt is valid (does not contain deny list entities),
            False otherwise.
            dict: The entities present in the prompt
        """
        prompt_payload = {"prompt": question}
        prompt_entities: dict = {"entities": {}, "entityCount": 0}
        is_valid_prompt: bool = True
        if self.classifier_location == "local":
            headers = self._make_headers()
            prompt_gov_api_url = (
                f"{self.classifier_url}{Routes.prompt_governance.value}"
            )
            pebblo_resp = self.make_request(
                "POST", prompt_gov_api_url, headers, prompt_payload
            )
            if pebblo_resp:
                prompt_entities["entities"] = pebblo_resp.json().get("entities", {})
                prompt_entities["entityCount"] = pebblo_resp.json().get(
                    "entityCount", 0
                )
        return is_valid_prompt, prompt_entities

    async def acheck_prompt_validity(
        self, question: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check the validity of the given prompt using a remote classification service.

        This method sends a prompt to a remote classifier service and return entities
        present in prompt or not.

        Args:
            question (str): The prompt question to be validated.

        Returns:
            bool: True if the prompt is valid (does not contain deny list entities),
            False otherwise.
            dict: The entities present in the prompt
        """
        prompt_payload = {"prompt": question}
        prompt_entities: dict = {"entities": {}, "entityCount": 0}
        is_valid_prompt: bool = True
        if self.classifier_location == "local":
            headers = self._make_headers()
            prompt_gov_api_url = (
                f"{self.classifier_url}{Routes.prompt_governance.value}"
            )
            pebblo_resp = await self.amake_request(
                "POST", prompt_gov_api_url, headers, prompt_payload
            )
            if pebblo_resp:
                prompt_entities["entities"] = pebblo_resp.get("entities", {})
                prompt_entities["entityCount"] = pebblo_resp.get("entityCount", 0)
        return is_valid_prompt, prompt_entities

    def is_privileged_user(self, auth_context: Optional[AuthContext]) -> bool:
        """
        Check if the user is a privileged user.

        Args:
            auth_context (Optional[AuthContext]): Authentication context.

        Returns:
            bool: True if the user is a privileged user, False otherwise.
        """
        if not auth_context or not self.policy_cache:
            return False

        # Get privileged_identities from the policy
        privileged_identities = self.policy_cache.privileged_identities
        if not privileged_identities:
            logger.debug("Privileged identities not found in the policy.")
            return False

        # Check if user is a privileged user
        user_auth = auth_context.user_auth if auth_context.user_auth else []
        is_privileged_user = any(
            _identity in privileged_identities for _identity in user_auth
        )
        if is_privileged_user:
            logger.debug(f"User {auth_context.user_id} is a privileged user.")
        else:
            logger.debug(f"User {auth_context.user_id} is not a privileged user.")
        return is_privileged_user

    def get_semantic_context(
        self, auth_context: Optional[AuthContext]
    ) -> Optional[SemanticContext]:
        """
        Generate semantic context based on the given auth context.

        Args:
            auth_context (Optional[AuthContext]): Authentication context.

        Returns:
            Optional[SemanticContext]: Semantic context.
        """
        semantic_context = None
        if not auth_context or not self.policy_cache:
            return semantic_context

        user_semantic_guardrail = self.policy_cache.user_semantic_guardrail
        if not user_semantic_guardrail:
            return semantic_context

        _all_guardrails = []
        user_auth = auth_context.user_auth if auth_context.user_auth else []
        for identity in user_auth:
            if guardrail := self.policy_cache.user_semantic_guardrail.get(identity):
                _all_guardrails.append(guardrail)

        semantic_context = self._combine_all_semantic_filters(_all_guardrails)
        return semantic_context

    def _start_policy_refresh_thread(self) -> None:
        """Start a thread to fetch policy from the Pebblo cloud."""
        logger.info("Starting policy refresh thread.")
        policy_thread = threading.Thread(target=self._fetch_policy, daemon=True)
        policy_thread.start()

    def _fetch_policy(self) -> None:
        """Fetch policy from the Pebblo cloud at regular intervals."""
        while True:
            try:
                # Fetch identity policy from the Pebblo cloud
                resp_json = self._get_policy_from_cloud(
                    self.app_name, PolicyType.IDENTITY
                )
                # Update the local cache with the fetched policy
                if resp_json:
                    policy_type = resp_json.get("type")
                    if not policy_type:
                        logger.debug(f"Message: {resp_json.get('message')}")
                        self.policy_cache = None
                    elif policy_type == PolicyType.IDENTITY.value:
                        self.policy_cache = IdentityPolicy(**resp_json)
                        logger.debug(f"Policy cache updated: {self.policy_cache}")
                    else:
                        logger.warning(f"Policy type {policy_type} not supported.")
            except Exception as e:
                logger.warning(f"Failed to fetch policy: {e}")
            # Sleep for the refresh interval
            time.sleep(_POLICY_REFRESH_INTERVAL_SEC)

    def _get_policy_from_cloud(self, app_name: str, policy_type: PolicyType) -> Any:
        """
        Get the policy for an app from the Pebblo Cloud.

        Args:
            app_name (str): Name of the app.
            policy_type (PolicyType): Type of policy to fetch.

        Returns:
            Any: Json response from the Pebblo Cloud.

        """
        resp_json = None
        policy_url = f"{self.cloud_url}{Routes.policy.value}"
        headers = self._make_headers(cloud_request=True)
        payload = {"app_name": app_name, "policy_type": policy_type.value}
        response = self.make_request("POST", policy_url, headers, payload)
        if response and response.status_code == HTTPStatus.OK:
            resp_json = response.json()
        else:
            logger.warning(f"Failed to fetch policy for {app_name}")
        return resp_json

    @staticmethod
    def _combine_all_semantic_filters(
        all_guardrails: Optional[list] = None,
    ) -> Optional[SemanticContext]:
        """
        Combine all provided guardrails to create a semantic context by finding the
        intersection of all guardrails.

        Args:
            all_guardrails (Optional[list]): List of guardrails.

        Returns:
            Optional[SemanticContext]: The generated semantic context.
        """
        if not all_guardrails:
            return None

        # Find the intersection of all guardrails
        entities_to_deny = set.intersection(
            *[_guardrail.entities for _guardrail in all_guardrails]
        )
        topics_to_deny = set.intersection(
            *[_guardrail.topics for _guardrail in all_guardrails]
        )

        # Generate semantic context from the deny entities and topics
        _semantic_context = dict()
        _semantic_context["pebblo_semantic_entities"] = {"deny": entities_to_deny}
        _semantic_context["pebblo_semantic_topics"] = {"deny": topics_to_deny}
        semantic_context = SemanticContext(**_semantic_context)
        return semantic_context

    def _make_headers(self, cloud_request: bool = False) -> dict:
        """
        Generate headers for the request.

        args:
            cloud_request (bool): flag indicating whether the request is for Pebblo
            cloud.
        returns:
            dict: Headers for the request.

        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if cloud_request:
            # Add API key for Pebblo cloud request
            if self.api_key:
                headers.update({"x-api-key": self.api_key})
            else:
                logger.warning("API key is missing for Pebblo cloud request.")
        return headers

    @staticmethod
    def make_request(
        method: str,
        url: str,
        headers: dict,
        payload: Optional[dict] = None,
        timeout: int = 20,
    ) -> Optional[Response]:
        """
        Make a request to the Pebblo server/cloud API.

        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.).
            url (str): URL for the request.
            headers (dict): Headers for the request.
            payload (Optional[dict]): Payload for the request (for POST, PUT, etc.).
            timeout (int): Timeout for the request in seconds.

        Returns:
            Optional[Response]: Response object if the request is successful.
        """
        try:
            response = request(
                method=method, url=url, headers=headers, json=payload, timeout=timeout
            )
            logger.debug(
                "Request: method %s, url %s, len %s response status %s",
                method,
                response.request.url,
                str(len(response.request.body if response.request.body else [])),
                str(response.status_code),
            )

            if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
                logger.warning(f"Pebblo Server: Error {response.status_code}")
            elif response.status_code >= HTTPStatus.BAD_REQUEST:
                logger.warning(f"Pebblo received an invalid payload: {response.text}")
            elif response.status_code != HTTPStatus.OK:
                logger.warning(
                    f"Pebblo returned an unexpected response code: "
                    f"{response.status_code}"
                )

            return response
        except RequestException:
            logger.warning("Unable to reach server %s", url)
        except Exception as e:
            logger.warning("An Exception caught in make_request: %s", e)
        return None

    @staticmethod
    def update_cloud_payload(payload: dict, pebblo_resp: Optional[dict]) -> None:
        """
        Update the payload with response, prompt and context from Pebblo response.

        Args:
            payload (dict): Payload to be updated.
            pebblo_resp (Optional[dict]): Response from Pebblo server.
        """
        if pebblo_resp:
            # Update response, prompt and context from pebblo response
            response = payload.get("response", {})
            response.update(pebblo_resp.get("retrieval_data", {}).get("response", {}))
            response.pop("data", None)
            prompt = payload.get("prompt", {})
            prompt.update(pebblo_resp.get("retrieval_data", {}).get("prompt", {}))
            prompt.pop("data", None)
            context = payload.get("context", [])
            for context_data in context:
                context_data.pop("doc", None)
        else:
            payload["response"] = {}
            payload["prompt"] = {}
            payload["context"] = []

    @staticmethod
    async def amake_request(
        method: str,
        url: str,
        headers: dict,
        payload: Optional[dict] = None,
        timeout: int = 20,
    ) -> Any:
        """
        Make an async request to the Pebblo server/cloud API.

        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.).
            url (str): URL for the request.
            headers (dict): Headers for the request.
            payload (Optional[dict]): Payload for the request (for POST, PUT, etc.).
            timeout (int): Timeout for the request in seconds.

        Returns:
            Any: Response json if the request is successful.
        """
        try:
            client_timeout = ClientTimeout(total=timeout)
            async with aiohttp.ClientSession() as asession:
                async with asession.request(
                    method=method,
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=client_timeout,
                ) as response:
                    if response.status >= HTTPStatus.INTERNAL_SERVER_ERROR:
                        logger.warning(f"Pebblo Server: Error {response.status}")
                    elif response.status >= HTTPStatus.BAD_REQUEST:
                        logger.warning(
                            f"Pebblo received an invalid payload: " f"{response.text}"
                        )
                    elif response.status != HTTPStatus.OK:
                        logger.warning(
                            f"Pebblo returned an unexpected response code: "
                            f"{response.status}"
                        )
                    response_json = await response.json()
            return response_json
        except RequestException:
            logger.warning("Unable to reach server %s", url)
        except Exception as e:
            logger.warning("An Exception caught in amake_request: %s", e)
        return None

    def build_prompt_qa_payload(
        self,
        app_name: str,
        retriever: VectorStoreRetriever,
        question: str,
        answer: str,
        auth_context: Optional[AuthContext],
        docs: List[Document],
        prompt_entities: Dict[str, Any],
        prompt_time: str,
        prompt_gov_enabled: bool = False,
    ) -> dict:
        """
        Build the QA payload for the prompt.

         Args:
            app_name (str): Name of the app.
            retriever (VectorStoreRetriever): Retriever instance.
            question (str): Question asked in the prompt.
            answer (str): Answer generated by the model.
            auth_context (Optional[AuthContext]): Authentication context.
            docs (List[Document]): List of documents retrieved.
            prompt_entities (Dict[str, Any]): Entities present in the prompt.
            prompt_time (str): Time when the prompt was generated.
            prompt_gov_enabled (bool): Whether prompt governance is enabled.

        Returns:
            dict: The QA payload for the prompt.
        """
        qa = Qa(
            name=app_name,
            context=[
                Context(
                    retrieved_from=doc.metadata.get(
                        "full_path", doc.metadata.get("source")
                    ),
                    doc=doc.page_content,
                    vector_db=retriever.vectorstore.__class__.__name__,
                    pb_checksum=doc.metadata.get("pb_checksum"),
                )
                for doc in docs
                if isinstance(doc, Document)
            ],
            prompt=Prompt(
                data=question,
                entities=prompt_entities.get("entities", {}),
                entityCount=prompt_entities.get("entityCount", 0),
                prompt_gov_enabled=prompt_gov_enabled,
            ),
            response=Prompt(data=answer),
            prompt_time=prompt_time,
            user=auth_context.user_id if auth_context else "unknown",
            user_identities=auth_context.user_auth
            if auth_context and hasattr(auth_context, "user_auth")
            else [],
            classifier_location=self.classifier_location,
        )
        return qa.model_dump(exclude_unset=True)
