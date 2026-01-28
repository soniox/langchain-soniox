import asyncio
import os
import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env

from .errors import (
    SonioxAPIError,
    SonioxClientError,
    SonioxTimeoutError,
    SonioxTranscriptionFailedError,
)
from .types import (
    SonioxCreateTranscriptionRequest,
    SonioxCreateTranscriptionResponse,
    SonioxFileUploadResponse,
    SonioxTranscriptionOptions,
    SonioxTranscriptionStatusResponse,
    SonioxTranscriptResponse,
)


class SonioxDocumentLoader(BaseLoader):
    """Loader for Soniox speech-to-text transcriptions.

    It uploads an audio file to Soniox, creates a transcription job,
    polls for its completion, and returns the transcript as a Document.
    """

    def __init__(
        self,
        *,
        file_path: Optional[str] = None,
        file_data: Optional[bytes] = None,
        file_url: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://api.soniox.com/v1",
        options: Optional[SonioxTranscriptionOptions] = None,
        polling_interval_seconds: float = 1.0,
        timeout_seconds: float = 5 * 60.0,
        http_request_timeout_seconds: float = 60.0,
    ):
        """Initialize with Soniox API key and transcription options.

        You must specify exactly one of: file_path, file_data, or file_url.

        Args:
            file_path: Optional path to the local audio file to transcribe.
            file_data: Optional binary data of the audio file to transcribe.
            file_url: Optional URL of the audio file to transcribe.
            api_key: Soniox API key. If not provided, looks for SONIOX_API_KEY in env.
            options: Optional transcription options, such as model, language hints, etc.
            polling_interval_seconds: Time in seconds between status polls.
            timeout_seconds: Maximum time in seconds to wait for transcription.
            http_request_timeout_seconds: Timeout for HTTP requests.
        """
        args_provided = sum(x is not None for x in (file_path, file_data, file_url))
        if args_provided != 1:
            raise SonioxClientError(
                "You must specify exactly one of 'file_path', 'file_data', or "
                "'file_url'."
            )

        self.api_key = api_key or get_from_env("api_key", "SONIOX_API_KEY")
        if not self.api_key:
            raise SonioxClientError(
                "Soniox API key must be provided or set in SONIOX_API_KEY env variable."
            )

        self.file_path = file_path
        self.file_data = file_data
        self.file_url = file_url

        self.polling_interval_seconds = polling_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url
        self.options = options if options else SonioxTranscriptionOptions()
        self.http_request_timeout_seconds = http_request_timeout_seconds

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _create_document(
        self,
        transcript_obj: SonioxTranscriptResponse,
        status_data: SonioxTranscriptionStatusResponse,
        transcription_id: str,
    ) -> Document:
        """Constructs the LangChain Document from API responses."""
        metadata = {
            "source": self.file_url or self.file_path or "file_upload",
            "transcription_id": transcription_id,
            "audio_duration_ms": status_data.audio_duration_ms,
            "model": status_data.model,
            "created_at": status_data.created_at,
            "tokens": [t.model_dump() for t in transcript_obj.tokens],
        }
        return Document(page_content=transcript_obj.text, metadata=metadata)

    def _prepare_create_payload(self, file_id: Optional[str] = None) -> Dict[str, Any]:
        """Prepares the payload for creating a transcription."""
        request_payload = SonioxCreateTranscriptionRequest(
            **self.options.model_dump(exclude_none=True),
        )

        if self.file_url:
            request_payload.audio_url = self.file_url
        elif file_id:
            request_payload.file_id = file_id
        else:
            raise SonioxClientError("No file source provided.")

        return request_payload.model_dump(exclude_none=True)

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Soniox transcriptions. Yields a single Document."""
        headers = self._get_headers()

        if self.base_url is None:
            raise SonioxClientError("base_url must be provided.")

        file_id: Optional[str] = None
        transcription_id: Optional[str] = None

        with httpx.Client(
            base_url=self.base_url,
            timeout=self.http_request_timeout_seconds,
        ) as client:
            try:
                # Upload file if needed
                if self.file_data:
                    files = {"file": ("audio_file", self.file_data)}
                    resp = client.post("/files", headers=headers, files=files)
                    if resp.is_error:
                        raise SonioxAPIError(resp)
                    file_id = SonioxFileUploadResponse.model_validate(resp.json()).id
                elif self.file_path:
                    with open(self.file_path, "rb") as f:
                        files = {"file": (os.path.basename(self.file_path), f)}
                        resp = client.post("/files", headers=headers, files=files)
                        if resp.is_error:
                            raise SonioxAPIError(resp)
                        file_id = SonioxFileUploadResponse.model_validate(
                            resp.json()
                        ).id
                else:
                    file_id = None

                # Create transcription
                payload = self._prepare_create_payload(file_id)
                resp = client.post("/transcriptions", headers=headers, json=payload)
                if resp.is_error:
                    raise SonioxAPIError(resp)
                transcription_id = SonioxCreateTranscriptionResponse.model_validate(
                    resp.json()
                ).id

                # Wait for transcription to complete
                start_time = time.time()
                final_status = None
                while True:
                    if time.time() - start_time > self.timeout_seconds:
                        raise SonioxTimeoutError(
                            f"Transcription timed out after {self.timeout_seconds}s"
                        )

                    resp = client.get(
                        f"/transcriptions/{transcription_id}", headers=headers
                    )
                    if resp.is_error:
                        raise SonioxAPIError(resp)
                    status_data = SonioxTranscriptionStatusResponse.model_validate(
                        resp.json()
                    )

                    if status_data.status == "completed":
                        final_status = status_data
                        break
                    elif status_data.status == "error":
                        raise SonioxTranscriptionFailedError(
                            f"Transcription failed: {status_data.error_message}"
                        )

                    time.sleep(self.polling_interval_seconds)

                # Get full transcript
                resp = client.get(
                    f"/transcriptions/{transcription_id}/transcript", headers=headers
                )
                if resp.is_error:
                    raise SonioxAPIError(resp)
                transcript_obj = SonioxTranscriptResponse.model_validate(resp.json())

                yield self._create_document(
                    transcript_obj, final_status, transcription_id
                )

            finally:
                # Best effort cleanup
                if transcription_id:
                    try:
                        client.delete(
                            f"/transcriptions/{transcription_id}", headers=headers
                        )
                    except Exception:
                        pass
                if file_id:
                    try:
                        client.delete(f"/files/{file_id}", headers=headers)
                    except Exception:
                        pass

    async def alazy_load(self) -> AsyncIterator[Document]:
        headers = self._get_headers()

        if self.base_url is None:
            raise SonioxClientError("base_url must be provided.")

        file_id: Optional[str] = None
        transcription_id: Optional[str] = None

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.http_request_timeout_seconds,
        ) as client:
            try:
                # Upload file if needed
                if self.file_data:
                    files = {"file": ("audio_file", self.file_data)}
                    resp = await client.post("/files", headers=headers, files=files)
                    if resp.is_error:
                        raise SonioxAPIError(resp)
                    file_id = SonioxFileUploadResponse.model_validate(resp.json()).id
                elif self.file_path:
                    # Note: httpx.AsyncClient requires standard file open to be blocking
                    # or use aiofiles. For simplicity without extra deps, we read into
                    # memory here.
                    with open(self.file_path, "rb") as f:
                        file_content = f.read()
                    files = {"file": (os.path.basename(self.file_path), file_content)}
                    resp = await client.post("/files", headers=headers, files=files)
                    if resp.is_error:
                        raise SonioxAPIError(resp)
                    file_id = SonioxFileUploadResponse.model_validate(resp.json()).id
                else:
                    file_id = None

                # Create transcription
                payload = self._prepare_create_payload(file_id)
                resp = await client.post(
                    "/transcriptions", headers=headers, json=payload
                )
                if resp.is_error:
                    raise SonioxAPIError(resp)
                transcription_id = SonioxCreateTranscriptionResponse.model_validate(
                    resp.json()
                ).id

                # Wait for transcription to complete
                start_time = time.time()
                final_status = None
                while True:
                    if time.time() - start_time > self.timeout_seconds:
                        raise SonioxTimeoutError(
                            f"Transcription timed out after {self.timeout_seconds}s"
                        )

                    resp = await client.get(
                        f"/transcriptions/{transcription_id}", headers=headers
                    )
                    if resp.is_error:
                        raise SonioxAPIError(resp)
                    status_data = SonioxTranscriptionStatusResponse.model_validate(
                        resp.json()
                    )

                    if status_data.status == "completed":
                        final_status = status_data
                        break
                    elif status_data.status == "error":
                        raise SonioxTranscriptionFailedError(
                            f"Transcription failed: {status_data.error_message}"
                        )

                    await asyncio.sleep(self.polling_interval_seconds)

                # Get full transcript
                resp = await client.get(
                    f"/transcriptions/{transcription_id}/transcript", headers=headers
                )
                if resp.is_error:
                    raise SonioxAPIError(resp)
                transcript_obj = SonioxTranscriptResponse.model_validate(resp.json())

                yield self._create_document(
                    transcript_obj, final_status, transcription_id
                )

            finally:
                # Best effort cleanup
                if transcription_id:
                    try:
                        await client.delete(
                            f"/transcriptions/{transcription_id}", headers=headers
                        )
                    except Exception:
                        pass
                if file_id:
                    try:
                        await client.delete(f"/files/{file_id}", headers=headers)
                    except Exception:
                        pass
