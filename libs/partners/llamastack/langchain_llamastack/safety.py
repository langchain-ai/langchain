"""LlamaStack Safety implementation for LangChain."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

try:
    from llama_stack_client import LlamaStackClient
except ImportError:
    LlamaStackClient = None


@dataclass
class SafetyResult:
    """Result from a safety check."""

    is_safe: bool
    message: str
    shield_id: str
    violation_type: Optional[str] = None
    confidence_score: float = 1.0
    raw_response: Dict[str, Any] = None

    def __post_init__(self):
        if self.raw_response is None:
            self.raw_response = {}


class LlamaStackSafety:
    """LlamaStack safety checking for content moderation."""

    def __init__(
        self,
        base_url: str = "http://localhost:8321",
        shield_id: Optional[str] = None,
        default_shield_model: Optional[str] = None,  # Deprecated, use shield_id
        llamastack_api_key: Optional[str] = None,
        safety_model_id: str = "ollama/llama-guard3:8b",
    ):
        """Initialize the LlamaStackSafety instance.

        Args:
            base_url: LlamaStack server URL
            shield_id: Specific shield model ID to use (e.g., 'llama-guard', 'code-scanner')
            default_shield_model: [DEPRECATED] Use shield_id instead
            llamastack_api_key: API key for LlamaStack authentication
            safety_model_id: Model ID to use for 'enabled' bug workaround (default: "ollama/llama-guard3:8b")

        Example:
            # Use specific shield model
            safety = LlamaStackSafety(shield_id="llama-guard")

            # Use specific shield with custom safety model
            safety = LlamaStackSafety(
                shield_id="llama-guard",
                safety_model_id="ollama/llama-guard3:8b"
            )

            # Auto-select available shield
            safety = LlamaStackSafety()
        """
        self.base_url = base_url
        self.llamastack_api_key = llamastack_api_key or ""
        self.safety_model_id = safety_model_id

        # Initialize LlamaStack client
        self.llamastack_available = False
        self.available_shields = []
        self.client = None

        if LlamaStackClient is None:
            raise ImportError(
                "llama-stack-client is required to use LlamaStackSafety. "
                "Install it with `pip install llama-stack-client`"
            )

        try:
            client_kwargs = {"base_url": base_url}
            if self.llamastack_api_key:
                client_kwargs["api_key"] = self.llamastack_api_key

            self.client = LlamaStackClient(**client_kwargs)
            self.available_shields = self._get_llamastack_shields()
            self.llamastack_available = len(self.available_shields) > 0

            if self.llamastack_available:
                logger.debug(f"LlamaStack shields available: {self.available_shields}")
            else:
                logger.warning("LlamaStack client created but no shields available")

        except Exception as e:
            logger.error(f"LlamaStack client initialization failed: {e}")
            raise ValueError(f"Cannot connect to LlamaStack at {base_url}: {e}")

        # Handle shield_id parameter (preferred) or fallback to deprecated default_shield_model
        self.user_specified_shield = shield_id or default_shield_model
        self.use_only_specified_shield = bool(shield_id or default_shield_model)

        if default_shield_model and not shield_id:
            logger.warning("default_shield_model is deprecated, use shield_id instead")

        # Set default shield model based on user preference
        if self.user_specified_shield:
            # User specified a particular shield - use it exclusively
            self.default_shield_model = self.user_specified_shield
            logger.info(
                f"Using user-specified shield model: {self.default_shield_model}"
            )

            # Warn if not in discovered shields, but still use it
            if self.user_specified_shield not in self.available_shields:
                available_shields_str = (
                    ", ".join(self.available_shields)
                    if self.available_shields
                    else "none"
                )
                logger.warning(
                    f"User-specified shield '{self.user_specified_shield}' not found in discovered shields. "
                    f"Will attempt to use it anyway. Discovered shields: {available_shields_str}"
                )
        elif self.available_shields:
            # No user preference - auto-select from available shields
            self.default_shield_model = self.available_shields[0]
            self.use_only_specified_shield = False
            logger.info(f"Auto-selected shield model: {self.default_shield_model}")
        else:
            # No shields available at all
            self.default_shield_model = None
            self.use_only_specified_shield = False
            logger.warning("No shield models found in LlamaStack")

    def _get_llamastack_shields(self) -> List[str]:
        """Get list of available shields from LlamaStack server."""
        try:
            if self.client is None:
                return []

            shields = self.client.shields.list()

            # Debug: Log the raw shield objects with more detail
            logger.debug(f"Raw shields response: {shields}")
            logger.debug(f"Shields type: {type(shields)}")

            shield_details = []
            for shield in shields:
                shield_dict = (
                    shield.__dict__ if hasattr(shield, "__dict__") else str(shield)
                )
                logger.debug(f"Shield object: {shield}")
                logger.debug(f"Shield dict: {shield_dict}")
                logger.debug(
                    f"Shield identifier: {getattr(shield, 'identifier', 'NO_IDENTIFIER')}"
                )

                # Check if shield has additional model mapping info
                for attr in ["model", "model_id", "model_name", "provider_resource_id"]:
                    if hasattr(shield, attr):
                        value = getattr(shield, attr)
                        logger.debug(f"Shield {attr}: {value}")
                        shield_details.append(f"{attr}={value}")

            shield_ids = [shield.identifier for shield in shields]

            # Filter out invalid shield identifiers
            valid_shield_ids = []
            for shield_id in shield_ids:
                # Skip generic status/config values that aren't actual model names
                if shield_id in ["enabled", "disabled", "true", "false", "on", "off"]:
                    logger.warning(f"Skipping invalid shield identifier: {shield_id}")
                    continue
                valid_shield_ids.append(shield_id)

            logger.debug(
                f"Found {len(valid_shield_ids)} valid shields in LlamaStack: {valid_shield_ids}"
            )
            return valid_shield_ids
        except Exception as e:
            logger.warning(f"Failed to get shields from LlamaStack: {e}")
            return []

    def get_available_shields(self) -> List[str]:
        """Get list of available shield models."""
        return self.available_shields

    def get_shield_info(self, shield_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about shields.

        Args:
            shield_id: Specific shield to get info for (optional)

        Returns:
            Dict containing shield information

        Example:
            .. code-block:: python

                # Get info for all shields
                info = safety.get_shield_info()

                # Get info for specific shield
                info = safety.get_shield_info("llama-guard")
        """
        try:
            if not self.client:
                return {"error": "LlamaStack client not available"}

            shields = self.client.shields.list()
            shields_info = {}

            for shield in shields:
                identifier = getattr(shield, "identifier", "unknown")

                # If specific shield requested, only return that one
                if shield_id and identifier != shield_id:
                    continue

                shield_data = {
                    "identifier": identifier,
                    "provider_id": getattr(shield, "provider_id", "unknown"),
                    "provider_resource_id": getattr(
                        shield, "provider_resource_id", "unknown"
                    ),
                    "type": getattr(shield, "type", "unknown"),
                    "params": getattr(shield, "params", {}),
                    "available": identifier in self.available_shields,
                }

                shields_info[identifier] = shield_data

            # If specific shield requested but not found
            if shield_id and shield_id not in shields_info:
                return {"error": f"Shield '{shield_id}' not found"}

            # If specific shield requested, return just that shield's info
            if shield_id:
                return shields_info[shield_id]

            # Return all shields info
            return {
                "total_shields": len(shields_info),
                "available_shields": len(self.available_shields),
                "shields": shields_info,
            }

        except Exception as e:
            return {"error": f"Failed to get shield info: {e}"}

    def check_content(
        self,
        content: str,
        shield_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> SafetyResult:
        """
        Check content for safety violations using LlamaStack.

        Args:
            content: Text content to check
            shield_id: Specific shield model to use (optional, overrides default)
            context: Additional context for the check (optional)

        Returns:
            SafetyResult: The safety check result

        Example:
            .. code-block:: python

                # Use default shield
                result = safety.check_content("Hello world!")
                if result.is_safe:
                    print("Content is safe")
                else:
                    print(f"Safety violation: {result.violation_type}")

                # Use specific shield
                result = safety.check_content("Hello world!", shield_id="llama_guard")
                print(f"Checked with {result.shield_id}: {'safe' if result.is_safe else 'unsafe'}")
        """
        model_to_use = shield_id or self.default_shield_model
        using_user_specified_shield = bool(shield_id) or self.use_only_specified_shield

        if not model_to_use:
            return SafetyResult(
                is_safe=True,
                message="No shield models available - content passed through",
                shield_id="",
                confidence_score=0.0,
            )

        # If user specified a shield (either in constructor or method call), use it regardless
        # of whether it's in available_shields. Only skip for auto-discovered shields.
        if (
            not using_user_specified_shield
            and model_to_use not in self.available_shields
        ):
            logger.warning(f"Auto-discovered shield model {model_to_use} not available")
            return SafetyResult(
                is_safe=True,
                message=f"Shield model {model_to_use} not available - content passed through",
                shield_id=model_to_use,
                confidence_score=0.0,
            )

        # Add context to content if provided
        content_to_check = content
        if context:
            content_to_check = f"Context: {context}\n\nContent: {content}"

        # Use LlamaStack only
        if not self.llamastack_available:
            raise ValueError(
                "LlamaStack is not available. Please ensure LlamaStack is running and accessible."
            )

        try:
            if using_user_specified_shield:
                logger.info(
                    f"Checking content with user-specified LlamaStack shield: {model_to_use}"
                )
            else:
                logger.info(
                    f"Checking content with auto-discovered LlamaStack shield: {model_to_use}"
                )
            return self._check_content_with_llamastack(content_to_check, model_to_use)
        except Exception as e:
            if using_user_specified_shield:
                logger.error(f"User-specified shield '{model_to_use}' failed: {e}")
                raise ValueError(
                    f"Failed to use user-specified shield '{model_to_use}': {e}"
                )
            else:
                logger.error(f"LlamaStack safety check failed: {e}")
                raise

    def _check_content_with_llamastack(
        self, content: str, shield_id: str
    ) -> SafetyResult:
        """Check content safety using LlamaStack server."""
        try:
            if not self.client:
                raise ValueError("LlamaStack client not available")

            logger.info(f"Checking content with shield: {shield_id}")
            logger.info(f"Content: '{content[:50]}...'")

            # Try normal shield call first
            response = self.client.safety.run_shield(
                shield_id=shield_id,
                messages=[{"role": "user", "content": content}],
                params={},  # Empty params dict as required parameter
            )

            # Process LlamaStack response - handle both formats
            is_safe = getattr(response, "is_safe", True)
            message = getattr(response, "message", "Content checked by LlamaStack")
            violation_type = getattr(response, "violation_type", None)

            # Check for violation object (alternative response format)
            violation = getattr(response, "violation", None)
            if violation:
                is_safe = False  # If violation exists, content is not safe
                violation_type = getattr(violation, "violation_type", "content_policy")
                user_message = getattr(violation, "user_message", str(violation))
                message = f"Safety violation detected: {user_message}"

            return SafetyResult(
                is_safe=is_safe,
                violation_type=violation_type,
                message=f"LlamaStack {shield_id}: {message}",
                shield_id=shield_id,
                raw_response=(
                    response.__dict__ if hasattr(response, "__dict__") else {}
                ),
            )

        except Exception as e:
            # Check if this is the 'enabled' bug and we can work around it
            if "Model 'enabled' not found" in str(e) and shield_id == "llama-guard":
                logger.info(
                    f"Shield '{shield_id}' has 'enabled' bug, trying direct model call workaround"
                )
                return self._call_model_for_safety_workaround(content, shield_id)
            else:
                logger.error(f"LlamaStack safety check failed: {e}")
                raise

    def _call_model_for_safety_workaround(
        self, content: str, shield_id: str
    ) -> SafetyResult:
        """Workaround for 'enabled' bug by calling safety model directly."""
        try:
            # Map shield to actual model
            # model_mapping = {"llama-guard": "ollama/llama-guard3:8b"}
            # model_mapping = {"llama-guard": "llama-guard"}

            # model_id = model_mapping.get(shield_id)
            # if not model_id:
            #     raise ValueError(f"No model mapping for shield '{shield_id}'")

            logger.info(
                f"WORKAROUND: Trying run_shield with model ID directly: {shield_id}"
            )

            # APPROACH 1: Try run_shield with shield_id (proper safety API)
            try:
                response = self.client.safety.run_shield(
                    shield_id=shield_id,
                    messages=[{"role": "user", "content": content}],
                    params={},
                )

                logger.info(f"WORKAROUND: run_shield with model ID succeeded!")

                # Process response using proper safety API response format
                is_safe = getattr(response, "is_safe", True)
                message = getattr(
                    response,
                    "message",
                    "Content checked via model-as-shield workaround",
                )
                violation_type = getattr(response, "violation_type", None)

                return SafetyResult(
                    is_safe=is_safe,
                    violation_type=violation_type,
                    message=f"Model-as-shield ({shield_id}): {message}",
                    shield_id=shield_id,  # Keep original shield_id for user reference
                    raw_response={"workaround": "model_as_shield", "model": shield_id},
                )

            except Exception as shield_error:
                logger.info(
                    f"WORKAROUND: run_shield with model ID failed: {shield_error}"
                )
                logger.info(
                    f"WORKAROUND: Falling back to inference API with {self.safety_model_id}"
                )

                # APPROACH 2: Fall back to inference API (less ideal but should work)
                safety_prompt = f"""You are a content safety classifier. Analyze the following content and respond with ONLY "SAFE" or "UNSAFE" followed by a brief reason.

Content to analyze: {content}

Classification:"""

                response = self.client.inference.chat_completion(
                    model_id=self.safety_model_id,  # Use configurable model for workaround
                    messages=[{"role": "user", "content": safety_prompt}],
                )

                # Parse response
                completion_content = ""
                if hasattr(response, "completion") and hasattr(
                    response.completion, "content"
                ):
                    completion_content = response.completion.content
                elif hasattr(response, "choices") and response.choices:
                    completion_content = response.choices[0].message.content
                else:
                    completion_content = str(response)

                # Determine safety
                completion_lower = completion_content.lower()
                is_safe = (
                    "safe" in completion_lower and "unsafe" not in completion_lower
                )

                logger.info(f"WORKAROUND: Inference API result: {completion_content}")

                return SafetyResult(
                    is_safe=is_safe,
                    violation_type=None if is_safe else "content_policy",
                    message=f"Inference workaround (ollama/llama-guard3:8b): {completion_content}",
                    shield_id=shield_id,
                    raw_response={
                        "workaround": "inference_api",
                        "model": "ollama/llama-guard3:8b",
                        "response": completion_content,
                    },
                )

        except Exception as e:
            logger.error(f"WORKAROUND: All workaround approaches failed: {e}")
            # Return safe default to avoid blocking content
            return SafetyResult(
                is_safe=True,
                message=f"All workarounds failed, defaulting to safe: {e}",
                shield_id=shield_id,
                confidence_score=0.1,
            )

    def check_conversation(
        self,
        messages: List[Dict[str, str]],
        shield_id: Optional[str] = None,
    ) -> SafetyResult:
        """
        Check conversation for safety violations using LlamaStack.

        Args:
            messages: List of conversation messages with 'role' and 'content' keys
            shield_id: Specific shield model to use (optional)

        Returns:
            SafetyResult: The safety check result

        Example:
            .. code-block:: python

                messages = [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi! How can I help?"}
                ]
                result = safety.check_conversation(messages)
                print(f"Conversation safe: {result.is_safe}")
        """
        model_to_use = shield_id or self.default_shield_model
        using_user_specified_shield = bool(shield_id) or self.use_only_specified_shield

        if not model_to_use:
            return SafetyResult(
                is_safe=True,
                message="No shield models available - conversation passed through",
                shield_id="",
                confidence_score=0.0,
            )

        # If user specified a shield (either in constructor or method call), use it regardless
        # of whether it's in available_shields. Only skip for auto-discovered shields.
        if (
            not using_user_specified_shield
            and model_to_use not in self.available_shields
        ):
            logger.warning(f"Auto-discovered shield model {model_to_use} not available")
            return SafetyResult(
                is_safe=True,
                message=f"Shield model {model_to_use} not available - conversation passed through",
                shield_id=model_to_use,
                confidence_score=0.0,
            )

        # Use LlamaStack only
        if not self.llamastack_available:
            raise ValueError(
                "LlamaStack is not available. Please ensure LlamaStack is running and accessible."
            )

        try:
            if using_user_specified_shield:
                logger.info(
                    f"Checking conversation with user-specified LlamaStack shield: {model_to_use}"
                )
            else:
                logger.info(
                    f"Checking conversation with auto-discovered LlamaStack shield: {model_to_use}"
                )

            # Use LlamaStack safety endpoint with full conversation
            response = self.client.safety.run_shield(
                shield_id=model_to_use,
                messages=messages,
                params={},  # Empty params dict as required parameter
            )

            # Process LlamaStack response - handle both formats
            is_safe = getattr(response, "is_safe", True)
            message = getattr(response, "message", "Conversation checked by LlamaStack")
            violation_type = getattr(response, "violation_type", None)

            # Check for violation object (alternative response format)
            violation = getattr(response, "violation", None)
            if violation:
                is_safe = False  # If violation exists, content is not safe
                violation_type = getattr(violation, "violation_type", "content_policy")
                user_message = getattr(violation, "user_message", str(violation))
                message = f"Safety violation detected: {user_message}"

            return SafetyResult(
                is_safe=is_safe,
                violation_type=violation_type,
                message=f"LlamaStack {model_to_use}: {message}",
                shield_id=model_to_use,
                raw_response=(
                    response.__dict__ if hasattr(response, "__dict__") else {}
                ),
            )

        except Exception as e:
            # Check if this is the 'enabled' bug for conversation check too
            if "Model 'enabled' not found" in str(e) and model_to_use == "llama-guard":
                logger.info(f"Conversation shield '{model_to_use}' has 'enabled' bug")
                # For conversation, we'll just check the last user message as a fallback
                last_user_message = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user_message = msg.get("content", "")
                        break

                if last_user_message:
                    return self._call_model_for_safety_workaround(
                        last_user_message, model_to_use
                    )
                else:
                    return SafetyResult(
                        is_safe=True,
                        message="No user message found in conversation, defaulting to safe",
                        shield_id=model_to_use,
                        confidence_score=0.5,
                    )
            else:
                if using_user_specified_shield:
                    logger.error(f"User-specified shield '{model_to_use}' failed: {e}")
                    raise ValueError(
                        f"Failed to use user-specified shield '{model_to_use}': {e}"
                    )
                else:
                    logger.error(f"LlamaStack conversation safety check failed: {e}")
                    raise
