"""Base class for all loaders that uses O365 Package"""
from __future__ import annotations

from pathlib import Path
from langchain.document_loaders.base import BaseLoader
from typing import TYPE_CHECKING, List
from pydantic import BaseModel, BaseSettings, Field, FilePath, SecretStr

if TYPE_CHECKING:
    from O365 import Account


class _O365Settings(BaseSettings):
    client_id: str = Field(..., env="O365_CLIENT_ID")
    client_secret: SecretStr = Field(..., env="O365_CLIENT_SECRET")

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = ".env"


class _O365TokenStorage(BaseSettings):
    token_path: FilePath = Field(Path.home() / ".credentials" / "o365_token.txt")


class O365BaseLoader(BaseLoader, BaseModel):
    def _auth(
        self, settings: _O365Settings, scopes: List[str], auth_with_token: bool = False
    ) -> Account:
        """
        Authenticates the OneDrive API client using the specified
        authentication method and returns the Account object.

        Returns:
            Account: The authenticated Account object.
        """
        try:
            from O365 import FileSystemTokenBackend, Account
        except ImportError:
            raise ValueError(
                "O365 package not found, please install it with `pip install o365`"
            )
        if auth_with_token:
            token_storage = _O365TokenStorage()
            token_path = token_storage.token_path
            token_backend = FileSystemTokenBackend(
                token_path=token_path.parent, token_filename=token_path.name
            )
            account = Account(
                credentials=(
                    settings.client_id,
                    settings.client_secret.get_secret_value(),
                ),
                scopes=scopes,
                token_backend=token_backend,
                **{"raise_http_errors": False},
            )
        else:
            token_backend = FileSystemTokenBackend(
                token_path=Path.home() / ".credentials"
            )
            account = Account(
                credentials=(
                    settings.client_id,
                    settings.client_secret.get_secret_value(),
                ),
                scopes=scopes,
                token_backend=token_backend,
                **{"raise_http_errors": False},
            )
            # make the auth
            account.authenticate()
        return account
