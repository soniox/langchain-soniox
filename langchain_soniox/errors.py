import httpx
from pydantic import ValidationError
from pydantic import BaseModel, ConfigDict, Field


class ApiErrorValidationError(BaseModel):
    error_type: str
    location: str
    message: str


class ApiError(BaseModel):
    model_config = ConfigDict(extra="allow")

    status_code: int
    error_type: str
    message: str
    validation_errors: list[ApiErrorValidationError] = Field(default_factory=list)
    request_id: str


class SonioxError(Exception):
    """Base exception for all Soniox document loader errors."""

    pass


class SonioxAPIError(SonioxError):
    """Raised when the Soniox API returns a non-2xx response."""

    def __init__(self, response: httpx.Response):
        self.response = response
        self.status_code = response.status_code
        self.request_id: str | None = None
        self.api_error: ApiError | None = None

        # Try to parse the structured error
        try:
            payload = response.json()
            self.api_error = ApiError.model_validate(payload)
            self.request_id = self.api_error.request_id
            message = self.api_error.message
        except (ValueError, ValidationError):
            message = response.text or f"Request failed with status {self.status_code}"

        super().__init__(message)


class SonioxClientError(SonioxError):
    """Raised when client-side inputs (files/arguments) are invalid."""

    pass


class SonioxTranscriptionFailedError(SonioxError):
    """Raised when transcription status on Soniox API is "error"."""

    pass


class SonioxTimeoutError(SonioxError):
    """Raised when the Soniox API times out."""

    pass
