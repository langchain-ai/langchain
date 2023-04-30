

from langchain.document_loaders.base import BaseLoader
from pydantic import BaseModel, BaseSettings, Field, SecretStr, FilePath, DirectoryPath
from O365 import Account, FileSystemTokenBackend
from pathlib import Path
SCOPES = ["offline_access", "Files.Read.All"]

class _OneDriveSettings(BaseSettings):
    client_id: str = Field(..., env='O365_CLIENT_ID')
    client_secret: SecretStr = Field(..., env='O365_CLIENT_SECRET')

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = '.env'

class _OneDriveTokenStorage(BaseSettings):
    token_path: FilePath = Field(Path.home() / ".credentials" / "o365_token.txt")


class OneDriveLoader(BaseLoader, BaseModel):
    settings = _OneDriveSettings()
    def _auth(self, auth_with_token: bool = False):
        if auth_with_token:
            token_storage =  _OneDriveTokenStorage()
            token_path = token_storage.token_path
            token_backend = FileSystemTokenBackend(
                token_path=token_path.parent, token_filename=token_path.name
            )
            account = Account(
                credentials=(self.settings.client_id, self.settings.client_secret.get_secret_value()),
                scopes=SCOPES,
                token_backend=token_backend,
            )
        else:
            account = Account(
                credentials=(self.settings.client_id, self.settings.client_secret.get_secret_value()),
                scopes=SCOPES,
            )
            # make the auth
            account.authenticate()
        return account
    
    def load(self, auth_with_token: bool = False):
        account = self._auth(auth_with_token=auth_with_token)
        return account