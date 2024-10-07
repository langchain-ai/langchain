from typing import Any, Optional, Sequence

from langchain_core._api.deprecation import deprecated
from langchain_core.documents import BaseDocumentTransformer, Document

from langchain_community.utilities.vertexai import get_client_info


@deprecated(
    since="0.0.32",
    removal="1.0",
    alternative_import="langchain_google_community.DocAIParser",
)
class GoogleTranslateTransformer(BaseDocumentTransformer):
    """Translate text documents using Google Cloud Translation."""

    def __init__(
        self,
        project_id: str,
        *,
        location: str = "global",
        model_id: Optional[str] = None,
        glossary_id: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            project_id: Google Cloud Project ID.
            location: (Optional) Translate model location.
            model_id: (Optional) Translate model ID to use.
            glossary_id: (Optional) Translate glossary ID to use.
            api_endpoint: (Optional) Regional endpoint to use.
        """
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import translate
        except ImportError as exc:
            raise ImportError(
                "Install Google Cloud Translate to use this parser."
                "(pip install google-cloud-translate)"
            ) from exc

        self.project_id = project_id
        self.location = location
        self.model_id = model_id
        self.glossary_id = glossary_id

        self._client = translate.TranslationServiceClient(
            client_info=get_client_info("translate"),
            client_options=(
                ClientOptions(api_endpoint=api_endpoint) if api_endpoint else None
            ),
        )
        self._parent_path = self._client.common_location_path(project_id, location)
        # For some reason, there's no `model_path()` method for the client.
        self._model_path = (
            f"{self._parent_path}/models/{model_id}" if model_id else None
        )
        self._glossary_path = (
            self._client.glossary_path(project_id, location, glossary_id)
            if glossary_id
            else None
        )

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Translate text documents using Google Translate.

        Arguments:
            source_language_code: ISO 639 language code of the input document.
            target_language_code: ISO 639 language code of the output document.
                For supported languages, refer to:
                https://cloud.google.com/translate/docs/languages
            mime_type: (Optional) Media Type of input text.
                Options: `text/plain`, `text/html`
        """
        try:
            from google.cloud import translate
        except ImportError as exc:
            raise ImportError(
                "Install Google Cloud Translate to use this parser."
                "(pip install google-cloud-translate)"
            ) from exc

        response = self._client.translate_text(
            request=translate.TranslateTextRequest(
                contents=[doc.page_content for doc in documents],
                parent=self._parent_path,
                model=self._model_path,
                glossary_config=translate.TranslateTextGlossaryConfig(
                    glossary=self._glossary_path
                ),
                source_language_code=kwargs.get("source_language_code", None),
                target_language_code=kwargs.get("target_language_code"),
                mime_type=kwargs.get("mime_type", "text/plain"),
            )
        )

        # If using a glossary, the translations will be in `glossary_translations`.
        translations = response.glossary_translations or response.translations

        return [
            Document(
                page_content=translation.translated_text,
                metadata={
                    **doc.metadata,
                    "model": translation.model,
                    "detected_language_code": translation.detected_language_code,
                },
            )
            for doc, translation in zip(documents, translations)
        ]
