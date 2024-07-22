from typing import Any, Callable, Dict


class BaseModerationCallbackHandler:
    """Base class for moderation callback handlers."""

    def __init__(self) -> None:
        if (
            self._is_method_unchanged(
                BaseModerationCallbackHandler.on_after_pii, self.on_after_pii
            )
            and self._is_method_unchanged(
                BaseModerationCallbackHandler.on_after_toxicity, self.on_after_toxicity
            )
            and self._is_method_unchanged(
                BaseModerationCallbackHandler.on_after_prompt_safety,
                self.on_after_prompt_safety,
            )
        ):
            raise NotImplementedError(
                "Subclasses must override at least one of on_after_pii(), "
                "on_after_toxicity(), or on_after_prompt_safety() functions."
            )

    def _is_method_unchanged(
        self, base_method: Callable, derived_method: Callable
    ) -> bool:
        return base_method.__qualname__ == derived_method.__qualname__

    async def on_after_pii(
        self, moderation_beacon: Dict[str, Any], unique_id: str, **kwargs: Any
    ) -> None:
        """Run after PII validation is complete."""
        pass

    async def on_after_toxicity(
        self, moderation_beacon: Dict[str, Any], unique_id: str, **kwargs: Any
    ) -> None:
        """Run after Toxicity validation is complete."""
        pass

    async def on_after_prompt_safety(
        self, moderation_beacon: Dict[str, Any], unique_id: str, **kwargs: Any
    ) -> None:
        """Run after Prompt Safety validation is complete."""
        pass

    @property
    def pii_callback(self) -> bool:
        return (
            self.on_after_pii.__func__  # type: ignore
            is not BaseModerationCallbackHandler.on_after_pii
        )

    @property
    def toxicity_callback(self) -> bool:
        return (
            self.on_after_toxicity.__func__  # type: ignore
            is not BaseModerationCallbackHandler.on_after_toxicity
        )

    @property
    def prompt_safety_callback(self) -> bool:
        return (
            self.on_after_prompt_safety.__func__  # type: ignore
            is not BaseModerationCallbackHandler.on_after_prompt_safety
        )
