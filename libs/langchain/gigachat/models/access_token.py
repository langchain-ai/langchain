from langchain.pydantic_v1 import BaseModel


class AccessToken(BaseModel):
    """
    access_token: Сгенерированный Access Token
    expires_at: Unix-время завершения действия Access Token в миллисекундах
    """

    access_token: str
    expires_at: int
