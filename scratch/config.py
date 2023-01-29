import os

class config:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # OpenAI API Key
    # https://beta.openai.com/account/api-keys
    # https://beta.openai.com/docs/api-reference/authentication
    # https://beta.openai.com/docs/api-reference/authentication#authentication
    OPENAI_API_KEY : str

    def get_openai_api_key(self):
        return self.OPENAI_API_KEY