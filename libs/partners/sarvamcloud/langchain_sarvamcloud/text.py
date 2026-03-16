"""Sarvam AI text services: Translation, Transliteration, and Language Detection."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

_TRANSLATION_MODES = Literal[
    "formal", "modern-colloquial", "classic-colloquial", "code-mixed"
]
_OUTPUT_SCRIPTS = Literal["roman", "fully-native", "spoken-form-in-native"]
_NUMERALS_FORMATS = Literal["international", "native"]
_TRANSLATION_MODELS = Literal["sarvam-translate:v1", "mayura:v1"]

# BCP-47 codes for the 11 languages supported by transliteration and LID
_INDIC_LANGUAGE_CODES = frozenset(
    {
        "bn-IN",
        "en-IN",
        "gu-IN",
        "hi-IN",
        "kn-IN",
        "ml-IN",
        "mr-IN",
        "od-IN",
        "pa-IN",
        "ta-IN",
        "te-IN",
    }
)


class SarvamTranslator(BaseModel):
    """Sarvam AI translation service.

    Translates text across 22 Indian languages with rich style controls.
    Unique features include code-mixed output, spoken-form script variants,
    and gender-aware translations.

    Setup:
        Install `langchain-sarvamcloud` and set the environment variable:

        ```bash
        pip install -U langchain-sarvamcloud
        export SARVAM_API_KEY="your-api-key"
        ```

    Models:
        - `sarvam-translate:v1` — Supports up to 2000 characters.
        - `mayura:v1` — Supports up to 1000 chars, supports `source="auto"`.

    Example:
        ```python
        from langchain_sarvamcloud import SarvamTranslator

        translator = SarvamTranslator(model="sarvam-translate:v1")
        result = translator.translate(
            "Hello, how are you?",
            source_language_code="en-IN",
            target_language_code="hi-IN",
            mode="formal",
        )
        print(result["translated_text"])
        ```
    """

    model: _TRANSLATION_MODELS = "sarvam-translate:v1"
    """Translation model.

    - `sarvam-translate:v1`: Max 2000 chars.
    - `mayura:v1`: Max 1000 chars, supports `source_language_code="auto"`.
    """

    mode: _TRANSLATION_MODES = "formal"
    """Translation style.

    - `formal`: Formal register.
    - `modern-colloquial`: Modern conversational style.
    - `classic-colloquial`: Classical conversational style.
    - `code-mixed`: Mixed-language output (e.g. Hinglish).
    """

    speaker_gender: Literal["Male", "Female"] | None = None
    """Gender for gender-aware translations."""

    output_script: _OUTPUT_SCRIPTS | None = None
    """Output script format.

    - `roman`: Romanized script.
    - `fully-native`: Native script with formal style.
    - `spoken-form-in-native`: Spoken form in native script.
    """

    numerals_format: _NUMERALS_FORMATS = "international"
    """Numeral representation.

    - `international`: Use 0–9.
    - `native`: Use language-specific numerals (e.g. ०–९ for Hindi).
    """

    api_subscription_key: SecretStr | None = Field(default=None)
    """Sarvam API subscription key. Reads from `SARVAM_API_KEY`."""

    _client: Any = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize the Sarvam client."""
        import os  # noqa: PLC0415

        if self.api_subscription_key is None:
            key = os.environ.get("SARVAM_API_KEY")
            if key:
                from pydantic import SecretStr as _SecretStr  # noqa: PLC0415

                self.api_subscription_key = _SecretStr(key)
        try:
            from sarvamai import SarvamAI  # noqa: PLC0415

            key_val = (
                self.api_subscription_key.get_secret_value()
                if self.api_subscription_key
                else None
            )
            self._client = SarvamAI(api_subscription_key=key_val)
        except ImportError as exc:
            msg = (
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            )
            raise ImportError(msg) from exc
        return self

    def translate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
        mode: _TRANSLATION_MODES | None = None,
        speaker_gender: Literal["Male", "Female"] | None = None,
        output_script: _OUTPUT_SCRIPTS | None = None,
        numerals_format: _NUMERALS_FORMATS | None = None,
    ) -> dict[str, Any]:
        """Translate text between Indian languages.

        Args:
            text: Text to translate. Max 2000 chars for `sarvam-translate:v1`,
                max 1000 chars for `mayura:v1`.
            source_language_code: BCP-47 source language code (e.g. `en-IN`).
                Use `"auto"` with `mayura:v1` for auto-detection.
            target_language_code: BCP-47 target language code (e.g. `hi-IN`).
            mode: Translation style. Overrides instance default.
            speaker_gender: Gender for gender-aware output. Overrides instance
                default.
            output_script: Output script format. Overrides instance default.
            numerals_format: Numeral style. Overrides instance default.

        Returns:
            Dict with `translated_text`, `source_language_code`, `request_id`.
        """
        kwargs: dict[str, Any] = {
            "input": text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
            "model": self.model,
            "mode": mode or self.mode,
            "numerals_format": numerals_format or self.numerals_format,
        }
        gender = speaker_gender or self.speaker_gender
        if gender:
            kwargs["speaker_gender"] = gender
        script = output_script or self.output_script
        if script:
            kwargs["output_script"] = script

        response = self._client.text.translate(**kwargs)
        if not isinstance(response, dict):
            return response.model_dump()
        return response


class SarvamTransliterator(BaseModel):
    """Sarvam AI transliteration service.

    Converts text between scripts (e.g. Devanagari ↔ Roman) for 11 Indian
    languages. Supports spoken-form conversion and numeral format options.

    Note: Transliteration between two Indic scripts (e.g. Hindi → Bengali) is
    NOT supported. Use translation for cross-Indic language conversion.

    Supported languages (11): bn-IN, en-IN, gu-IN, hi-IN, kn-IN, ml-IN,
        mr-IN, od-IN, pa-IN, ta-IN, te-IN.

    Setup:
        Install `langchain-sarvamcloud` and set the environment variable:

        ```bash
        pip install -U langchain-sarvamcloud
        export SARVAM_API_KEY="your-api-key"
        ```

    Example:
        ```python
        from langchain_sarvamcloud import SarvamTransliterator

        tl = SarvamTransliterator()
        result = tl.transliterate(
            "नमस्ते",
            source_language_code="hi-IN",
            target_language_code="en-IN",
        )
        print(result["transliterated_text"])  # "namaste"
        ```
    """

    numerals_format: _NUMERALS_FORMATS = "international"
    """Numeral representation in output.

    - `international`: Use 0–9.
    - `native`: Use language-specific numerals.
    """

    spoken_form: bool = False
    """Convert written text to natural spoken form."""

    spoken_form_numerals_language: Literal["english", "native"] = "english"
    """Language for numeral pronunciation when `spoken_form=True`."""

    api_subscription_key: SecretStr | None = Field(default=None)
    """Sarvam API subscription key. Reads from `SARVAM_API_KEY`."""

    _client: Any = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize the Sarvam client."""
        import os  # noqa: PLC0415

        if self.api_subscription_key is None:
            key = os.environ.get("SARVAM_API_KEY")
            if key:
                from pydantic import SecretStr as _SecretStr  # noqa: PLC0415

                self.api_subscription_key = _SecretStr(key)
        try:
            from sarvamai import SarvamAI  # noqa: PLC0415

            key_val = (
                self.api_subscription_key.get_secret_value()
                if self.api_subscription_key
                else None
            )
            self._client = SarvamAI(api_subscription_key=key_val)
        except ImportError as exc:
            msg = (
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            )
            raise ImportError(msg) from exc
        return self

    def transliterate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
        spoken_form: bool | None = None,
        numerals_format: _NUMERALS_FORMATS | None = None,
        spoken_form_numerals_language: Literal["english", "native"] | None = None,
    ) -> dict[str, Any]:
        """Transliterate text between scripts.

        Args:
            text: Text to transliterate. Maximum 1000 characters.
            source_language_code: BCP-47 source language code.
            target_language_code: BCP-47 target language code.
            spoken_form: Convert to natural spoken form. Overrides instance
                default.
            numerals_format: Numeral style. Overrides instance default.
            spoken_form_numerals_language: Numeral pronunciation language.
                Overrides instance default.

        Returns:
            Dict with `transliterated_text`, `source_language_code`,
            `request_id`.

        Raises:
            ValueError: If both source and target are non-English Indic
                languages (Indic-to-Indic not supported).
        """
        non_english_source = source_language_code != "en-IN"
        non_english_target = target_language_code != "en-IN"
        if non_english_source and non_english_target:
            msg = (
                "Indic-to-Indic transliteration is not supported by Sarvam AI. "
                "Use SarvamTranslator for cross-Indic language conversion instead."
            )
            raise ValueError(msg)

        kwargs: dict[str, Any] = {
            "input": text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
            "spoken_form": (
                spoken_form if spoken_form is not None else self.spoken_form
            ),
            "numerals_format": numerals_format or self.numerals_format,
            "spoken_form_numerals_language": (
                spoken_form_numerals_language or self.spoken_form_numerals_language
            ),
        }
        response = self._client.text.transliterate(**kwargs)
        if not isinstance(response, dict):
            return response.model_dump()
        return response


class SarvamLanguageDetector(BaseModel):
    """Sarvam AI language identification service.

    Detects the language and script of input text across 11 Indian languages.
    Returns a language code, script code, and confidence score.

    Supported languages (11): en-IN, hi-IN, bn-IN, gu-IN, kn-IN, ml-IN,
        mr-IN, od-IN, pa-IN, ta-IN, te-IN.

    Supported scripts (10): Latin, Devanagari, Bengali, Gujarati, Kannada,
        Malayalam, Odia, Gurmukhi, Tamil, Telugu.

    Setup:
        Install `langchain-sarvamcloud` and set the environment variable:

        ```bash
        pip install -U langchain-sarvamcloud
        export SARVAM_API_KEY="your-api-key"
        ```

    Example:
        ```python
        from langchain_sarvamcloud import SarvamLanguageDetector

        detector = SarvamLanguageDetector()
        result = detector.detect("नमस्ते, आप कैसे हैं?")
        print(result["language_code"])   # "hi-IN"
        print(result["script_code"])     # "Deva"
        print(result["confidence"])      # 0.98
        ```
    """

    api_subscription_key: SecretStr | None = Field(default=None)
    """Sarvam API subscription key. Reads from `SARVAM_API_KEY`."""

    _client: Any = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize the Sarvam client."""
        import os  # noqa: PLC0415

        if self.api_subscription_key is None:
            key = os.environ.get("SARVAM_API_KEY")
            if key:
                from pydantic import SecretStr as _SecretStr  # noqa: PLC0415

                self.api_subscription_key = _SecretStr(key)
        try:
            from sarvamai import SarvamAI  # noqa: PLC0415

            key_val = (
                self.api_subscription_key.get_secret_value()
                if self.api_subscription_key
                else None
            )
            self._client = SarvamAI(api_subscription_key=key_val)
        except ImportError as exc:
            msg = (
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            )
            raise ImportError(msg) from exc
        return self

    def detect(self, text: str) -> dict[str, Any]:
        """Detect the language and script of text.

        Args:
            text: Input text to analyze. Maximum 1000 characters.

        Returns:
            Dict with nullable fields: `language_code` (BCP-47),
            `script_code` (ISO 15924), `confidence` (0.0–1.0),
            and non-nullable `request_id`.
        """
        response = self._client.text.identify_language(input=text)
        if not isinstance(response, dict):
            return response.model_dump()
        return response
