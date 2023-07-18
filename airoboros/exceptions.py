class RateLimitError(RuntimeError):
    ...


class TooManyRequestsError(RuntimeError):
    ...


class BadResponseError(RuntimeError):
    ...


class TokensExhaustedError(RuntimeError):
    ...


class ContextLengthExceededError(RuntimeError):
    ...


class ServerOverloadedError(RuntimeError):
    ...


class ServerError(RuntimeError):
    ...
