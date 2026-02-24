from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


class StructuredContextGeneralItem(BaseModel):
    key: str
    value: str


class StructuredContextTranslationTerm(BaseModel):
    source: str
    target: str


class StructuredContext(BaseModel):
    general: list[StructuredContextGeneralItem] | None = None
    text: str | None = None
    terms: list[str] | None = None
    translation_terms: list[StructuredContextTranslationTerm] | None = None


TranslationType = Literal["one_way", "two_way"]


class TranslationConfig(BaseModel):
    type: TranslationType
    target_language: str | None = None
    language_a: str | None = None
    language_b: str | None = None


class SonioxTranscriptionOptions(BaseModel):
    """
    Configuration parameters for a Soniox transcription job.

    Full details are available in the Soniox API documentation:
    https://soniox.com/docs/stt/api-reference/transcriptions/create_transcription
    """

    model: str = "stt-async-v4"
    language_hints: Optional[List[str]] = None
    language_hints_strict: Optional[bool] = None
    enable_speaker_diarization: Optional[bool] = None
    enable_language_identification: Optional[bool] = None
    translation: Optional[TranslationConfig] = None
    context: Optional[Union[StructuredContext, str]] = None
    webhook_url: Optional[str] = None
    webhook_auth_header_name: Optional[str] = None
    webhook_auth_header_value: Optional[str] = None
    client_reference_id: Optional[str] = None


class SonioxCreateTranscriptionRequest(SonioxTranscriptionOptions):
    audio_url: Optional[str] = None
    file_id: Optional[str] = None


class SonioxFileUploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    created_at: str
    client_reference_id: Optional[str] = None


TranscriptionStatus = Literal["queued", "processing", "completed", "error"]


class SonioxCreateTranscriptionResponse(BaseModel):
    id: str
    status: TranscriptionStatus
    created_at: str
    model: str
    filename: str
    enable_speaker_diarization: bool
    enable_language_identification: bool
    audio_url: Optional[str] = None
    file_id: Optional[str] = None
    language_hints: Optional[List[str]] = None
    audio_duration_ms: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_auth_header_name: Optional[str] = None
    webhook_auth_header_value: Optional[str] = None
    webhook_status_code: Optional[int] = None
    client_reference_id: Optional[str] = None


class SonioxTranscriptionStatusResponse(SonioxCreateTranscriptionResponse):
    pass


class TranscriptionToken(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    confidence: float
    speaker: Optional[str] = None
    language: Optional[str] = None
    translation_status: Optional[str] = None


class SonioxTranscriptResponse(BaseModel):
    id: str
    text: str
    tokens: List[TranscriptionToken]
