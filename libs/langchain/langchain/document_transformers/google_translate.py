from typing import Any, List, Optional, Sequence

from langchain.schema import BaseDocumentTransformer, Document
from langchain.utilities.vertexai import get_client_info


class GoogleTranslateTransformer(BaseDocumentTransformer):
    """Translate text documents using Google Cloud Translation.

    Arguments:
        project_id: Google Cloud Project ID.
        location: Translate model location.
        model_id: (Optional) Translate model ID to use.
        glossary_id: (Optional) Translate glossary ID to use.
        api_endpoint: (Optional) Regional endpoint to use.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "global",
        model_id: Optional[str] = None,
        glossary_id: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ) -> None:
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import translate
        except ImportError as exc:
            raise ImportError(
                "Install Google Translate to use this parser. (pip install google-cloud-translate)"
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
                "Install Google Translate to use this parser. (pip install google-cloud-translate)"
            ) from exc

        transformed_documents: List[Document] = []

        for doc in documents:
            response = self._client.translate_text(
                request=translate.TranslateTextRequest(
                    contents=[doc.page_content],
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
            new_metadata = {
                "model": getattr(translations[0], ("model")),
                "detected_language_code": getattr(
                    translations[0], "detected_language_code"
                ),
            }
            transformed_documents.append(
                Document(
                    page_content=translations[0].translated_text,
                    metadata={**doc.metadata, **new_metadata},
                )
            )
        return transformed_documents
