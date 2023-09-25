class GigaChatException(Exception):
    ...


class ResponseError(GigaChatException):
    ...


class AuthenticationError(ResponseError):
    ...
