from unittest.mock import MagicMock, patch

import pytest

from langchain_soniox import SonioxDocumentLoader
from langchain_soniox.errors import (
    SonioxAPIError,
    SonioxClientError,
    SonioxTimeoutError,
    SonioxTranscriptionFailedError,
)
from langchain_soniox.types import SonioxTranscriptionOptions


def test_no_input_provided():
    """Test that providing no input raises SonioxClientError."""
    with pytest.raises(
        SonioxClientError,
        match="You must specify exactly one of 'file_path', 'file_data', or 'file_url'",
    ):
        SonioxDocumentLoader(api_key="test_key")


def test_multiple_inputs_provided():
    """Test that providing multiple inputs raises SonioxClientError."""
    with pytest.raises(
        SonioxClientError,
        match="You must specify exactly one of 'file_path', 'file_data', or 'file_url'",
    ):
        SonioxDocumentLoader(
            file_path="test.mp3", file_data=b"data", api_key="test_key"
        )


def test_missing_api_key():
    """Test that missing API key raises ValueError from get_from_env."""
    with pytest.raises(ValueError, match="Did not find api_key"):
        SonioxDocumentLoader(file_path="test.mp3")


@patch.dict("os.environ", {"SONIOX_API_KEY": "env_key"})
def test_api_key_from_env():
    """Test that API key can be loaded from environment variable."""
    loader = SonioxDocumentLoader(file_path="test.mp3")
    assert loader.api_key == "env_key"


@patch("httpx.Client")
@patch("builtins.open", create=True)
def test_load_with_file_path(mock_open, mock_httpx_client):
    """Test successful loading with file_path."""
    mock_client = mock_httpx_client.return_value.__enter__.return_value

    # Setup mocks
    mock_upload_resp = MagicMock()
    mock_upload_resp.is_error = False
    mock_upload_resp.json.return_value = {
        "id": "file_123",
        "filename": "test.mp3",
        "size": 1024,
        "created_at": "2024-11-26T00:00:00Z",
    }

    mock_create_resp = MagicMock()
    mock_create_resp.is_error = False
    mock_create_resp.json.return_value = {
        "id": "trans_456",
        "status": "queued",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
    }

    mock_status_resp = MagicMock()
    mock_status_resp.is_error = False
    mock_status_resp.json.return_value = {
        "id": "trans_456",
        "status": "completed",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
        "audio_duration_ms": 1000,
    }

    mock_transcript_resp = MagicMock()
    mock_transcript_resp.is_error = False
    mock_transcript_resp.json.return_value = {
        "id": "trans_456",
        "text": "Hello world",
        "tokens": [
            {"text": "Hello", "start_ms": 0, "end_ms": 500, "confidence": 0.99},
            {
                "text": "world",
                "start_ms": 500,
                "end_ms": 1000,
                "confidence": 0.99,
            },
        ],
    }

    mock_client.post.side_effect = [mock_upload_resp, mock_create_resp]
    mock_client.get.side_effect = [mock_status_resp, mock_transcript_resp]
    mock_client.delete.return_value = MagicMock(is_error=False)

    loader = SonioxDocumentLoader(file_path="test.mp3", api_key="test_key")
    docs = list(loader.lazy_load())

    assert len(docs) == 1
    assert docs[0].page_content == "Hello world"
    assert docs[0].metadata["transcription_id"] == "trans_456"
    assert docs[0].metadata["source"] == "test.mp3"
    assert docs[0].metadata["audio_duration_ms"] == 1000
    assert docs[0].metadata["model"] == "stt-rt-v3"
    assert len(docs[0].metadata["tokens"]) == 2

    # Verify cleanup
    assert mock_client.delete.call_count == 2


@patch("httpx.Client")
def test_load_with_file_data(mock_httpx_client):
    """Test successful loading with file_data (bytes)."""
    mock_client = mock_httpx_client.return_value.__enter__.return_value

    # Setup mocks (similar to file_path test)
    mock_upload_resp = MagicMock()
    mock_upload_resp.is_error = False
    mock_upload_resp.json.return_value = {
        "id": "file_789",
        "filename": "audio_file",
        "size": 2048,
        "created_at": "2024-11-26T00:00:00Z",
    }

    mock_create_resp = MagicMock()
    mock_create_resp.is_error = False
    mock_create_resp.json.return_value = {
        "id": "trans_999",
        "status": "queued",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "audio_file",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
    }

    mock_status_resp = MagicMock()
    mock_status_resp.is_error = False
    mock_status_resp.json.return_value = {
        "id": "trans_999",
        "status": "completed",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "audio_file",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
        "audio_duration_ms": 2000,
    }

    mock_transcript_resp = MagicMock()
    mock_transcript_resp.is_error = False
    mock_transcript_resp.json.return_value = {
        "id": "trans_999",
        "text": "Test transcript",
        "tokens": [
            {"text": "Test", "start_ms": 0, "end_ms": 1000, "confidence": 0.95},
            {
                "text": "transcript",
                "start_ms": 1000,
                "end_ms": 2000,
                "confidence": 0.98,
            },
        ],
    }

    mock_client.post.side_effect = [mock_upload_resp, mock_create_resp]
    mock_client.get.side_effect = [mock_status_resp, mock_transcript_resp]
    mock_client.delete.return_value = MagicMock(is_error=False)

    loader = SonioxDocumentLoader(file_data=b"fake_audio_data", api_key="test_key")
    docs = list(loader.lazy_load())

    assert len(docs) == 1
    assert docs[0].page_content == "Test transcript"
    assert docs[0].metadata["source"] == "file_upload"
    assert mock_client.delete.call_count == 2


@patch("httpx.Client")
def test_load_with_file_url(mock_httpx_client):
    """Test successful loading with file_url (no upload needed)."""
    mock_client = mock_httpx_client.return_value.__enter__.return_value

    # No upload response needed for URL
    mock_create_resp = MagicMock()
    mock_create_resp.is_error = False
    mock_create_resp.json.return_value = {
        "id": "trans_url_123",
        "status": "queued",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "remote.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
    }

    mock_status_resp = MagicMock()
    mock_status_resp.is_error = False
    mock_status_resp.json.return_value = {
        "id": "trans_url_123",
        "status": "completed",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "remote.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
        "audio_duration_ms": 3000,
    }

    mock_transcript_resp = MagicMock()
    mock_transcript_resp.is_error = False
    mock_transcript_resp.json.return_value = {
        "id": "trans_url_123",
        "text": "Remote audio",
        "tokens": [
            {"text": "Remote", "start_ms": 0, "end_ms": 1500, "confidence": 0.99},
            {"text": "audio", "start_ms": 1500, "end_ms": 3000, "confidence": 0.97},
        ],
    }

    mock_client.post.side_effect = [mock_create_resp]
    mock_client.get.side_effect = [mock_status_resp, mock_transcript_resp]
    mock_client.delete.return_value = MagicMock(is_error=False)

    loader = SonioxDocumentLoader(
        file_url="https://example.com/audio.mp3", api_key="test_key"
    )
    docs = list(loader.lazy_load())

    assert len(docs) == 1
    assert docs[0].page_content == "Remote audio"
    assert docs[0].metadata["source"] == "https://example.com/audio.mp3"
    # Only transcription cleanup, no file upload
    assert mock_client.delete.call_count == 1


@patch("httpx.Client")
@patch("builtins.open", create=True)
def test_upload_api_error(mock_open, mock_httpx_client):
    """Test that upload API errors are properly raised."""
    mock_client = mock_httpx_client.return_value.__enter__.return_value

    mock_upload_resp = MagicMock()
    mock_upload_resp.is_error = True
    mock_upload_resp.status_code = 500
    mock_upload_resp.text = "Internal Server Error"

    mock_client.post.return_value = mock_upload_resp

    loader = SonioxDocumentLoader(file_path="test.mp3", api_key="test_key")

    with pytest.raises(SonioxAPIError):
        list(loader.lazy_load())


@patch("httpx.Client")
@patch("builtins.open", create=True)
def test_transcription_failed_error(mock_open, mock_httpx_client):
    """Test that transcription failure status raises proper error."""
    mock_client = mock_httpx_client.return_value.__enter__.return_value

    mock_upload_resp = MagicMock()
    mock_upload_resp.is_error = False
    mock_upload_resp.json.return_value = {
        "id": "file_123",
        "filename": "test.mp3",
        "size": 1024,
        "created_at": "2024-11-26T00:00:00Z",
    }

    mock_create_resp = MagicMock()
    mock_create_resp.is_error = False
    mock_create_resp.json.return_value = {
        "id": "trans_456",
        "status": "queued",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
    }

    mock_status_resp = MagicMock()
    mock_status_resp.is_error = False
    mock_status_resp.json.return_value = {
        "id": "trans_456",
        "status": "error",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
        "audio_duration_ms": 0,
        "error_message": "Audio format not supported",
    }

    mock_client.post.side_effect = [mock_upload_resp, mock_create_resp]
    mock_client.get.return_value = mock_status_resp
    mock_client.delete.return_value = MagicMock(is_error=False)

    loader = SonioxDocumentLoader(file_path="test.mp3", api_key="test_key")

    with pytest.raises(
        SonioxTranscriptionFailedError, match="Audio format not supported"
    ):
        list(loader.lazy_load())

    # Verify cleanup still happens
    assert mock_client.delete.call_count == 2


@patch("httpx.Client")
@patch("builtins.open", create=True)
@patch("time.time")
def test_transcription_timeout(mock_time, mock_open, mock_httpx_client):
    """Test that timeout is properly raised."""
    mock_client = mock_httpx_client.return_value.__enter__.return_value

    # Simulate time passing
    mock_time.side_effect = [0, 0, 10, 20, 30, 40]  # Exceeds 5 second timeout

    mock_upload_resp = MagicMock()
    mock_upload_resp.is_error = False
    mock_upload_resp.json.return_value = {
        "id": "file_123",
        "filename": "test.mp3",
        "size": 1024,
        "created_at": "2024-11-26T00:00:00Z",
    }

    mock_create_resp = MagicMock()
    mock_create_resp.is_error = False
    mock_create_resp.json.return_value = {
        "id": "trans_456",
        "status": "queued",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
    }

    # Always return "processing" status
    mock_status_resp = MagicMock()
    mock_status_resp.is_error = False
    mock_status_resp.json.return_value = {
        "id": "trans_456",
        "status": "processing",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
        "audio_duration_ms": 0,
    }

    mock_client.post.side_effect = [mock_upload_resp, mock_create_resp]
    mock_client.get.return_value = mock_status_resp
    mock_client.delete.return_value = MagicMock(is_error=False)

    loader = SonioxDocumentLoader(
        file_path="test.mp3",
        api_key="test_key",
        timeout_seconds=5.0,
        polling_interval_seconds=0.1,
    )

    with pytest.raises(SonioxTimeoutError, match="timed out after 5.0s"):
        list(loader.lazy_load())

    # Verify cleanup still happens
    assert mock_client.delete.call_count == 2


@patch("httpx.Client")
def test_custom_transcription_options(mock_httpx_client):
    """Test that custom transcription options are passed correctly."""
    mock_client = mock_httpx_client.return_value.__enter__.return_value

    mock_create_resp = MagicMock()
    mock_create_resp.is_error = False
    mock_create_resp.json.return_value = {
        "id": "trans_456",
        "status": "queued",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "remote.mp3",
        "enable_speaker_diarization": True,
        "enable_language_identification": True,
    }

    mock_status_resp = MagicMock()
    mock_status_resp.is_error = False
    mock_status_resp.json.return_value = {
        "id": "trans_456",
        "status": "completed",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "remote.mp3",
        "enable_speaker_diarization": True,
        "enable_language_identification": True,
        "audio_duration_ms": 1000,
    }

    mock_transcript_resp = MagicMock()
    mock_transcript_resp.is_error = False
    mock_transcript_resp.json.return_value = {
        "id": "trans_456",
        "text": "Test",
        "tokens": [{"text": "Test", "start_ms": 0, "end_ms": 1000, "confidence": 0.99}],
    }

    mock_client.post.side_effect = [mock_create_resp]
    mock_client.get.side_effect = [mock_status_resp, mock_transcript_resp]
    mock_client.delete.return_value = MagicMock(is_error=False)

    options = SonioxTranscriptionOptions(
        model="stt-rt-v3",
        enable_speaker_diarization=True,
        enable_language_identification=True,
    )

    loader = SonioxDocumentLoader(
        file_url="https://example.com/audio.mp3", api_key="test_key", options=options
    )
    list(loader.lazy_load())

    # Verify the payload sent to create transcription
    call_args = mock_client.post.call_args_list[0]
    payload = call_args[1]["json"]
    assert payload["model"] == "stt-rt-v3"
    assert payload["enable_speaker_diarization"] is True
    assert payload["enable_language_identification"] is True


@patch("httpx.AsyncClient")
@patch("builtins.open", create=True)
async def test_async_load_success(mock_open, mock_httpx_async_client):
    """Test successful async loading."""
    mock_client = mock_httpx_async_client.return_value.__aenter__.return_value

    mock_upload_resp = MagicMock()
    mock_upload_resp.is_error = False
    mock_upload_resp.json.return_value = {
        "id": "file_123",
        "filename": "test.mp3",
        "size": 1024,
        "created_at": "2024-11-26T00:00:00Z",
    }

    mock_create_resp = MagicMock()
    mock_create_resp.is_error = False
    mock_create_resp.json.return_value = {
        "id": "trans_456",
        "status": "queued",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
    }

    mock_status_resp = MagicMock()
    mock_status_resp.is_error = False
    mock_status_resp.json.return_value = {
        "id": "trans_456",
        "status": "completed",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
        "audio_duration_ms": 1000,
    }

    mock_transcript_resp = MagicMock()
    mock_transcript_resp.is_error = False
    mock_transcript_resp.json.return_value = {
        "id": "trans_456",
        "text": "Hello world",
        "tokens": [
            {"text": "Hello", "start_ms": 0, "end_ms": 500, "confidence": 0.99},
            {
                "text": "world",
                "start_ms": 500,
                "end_ms": 1000,
                "confidence": 0.99,
            },
        ],
    }

    mock_client.post.side_effect = [mock_upload_resp, mock_create_resp]
    mock_client.get.side_effect = [mock_status_resp, mock_transcript_resp]
    mock_client.delete.return_value = MagicMock(is_error=False)

    loader = SonioxDocumentLoader(file_path="test.mp3", api_key="test_key")
    docs = [doc async for doc in loader.alazy_load()]

    assert len(docs) == 1
    assert docs[0].page_content == "Hello world"
    assert docs[0].metadata["transcription_id"] == "trans_456"
    assert docs[0].metadata["source"] == "test.mp3"

    assert mock_client.delete.call_count == 2


@patch("httpx.AsyncClient")
@patch("builtins.open", create=True)
async def test_async_transcription_failed_error(mock_open, mock_httpx_async_client):
    """Test async error handling for failed transcription."""
    mock_client = mock_httpx_async_client.return_value.__aenter__.return_value

    mock_upload_resp = MagicMock()
    mock_upload_resp.is_error = False
    mock_upload_resp.json.return_value = {
        "id": "file_123",
        "filename": "test.mp3",
        "size": 1024,
        "created_at": "2024-11-26T00:00:00Z",
    }

    mock_create_resp = MagicMock()
    mock_create_resp.is_error = False
    mock_create_resp.json.return_value = {
        "id": "trans_456",
        "status": "queued",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
    }

    mock_status_resp = MagicMock()
    mock_status_resp.is_error = False
    mock_status_resp.json.return_value = {
        "id": "trans_456",
        "status": "error",
        "created_at": "2024-11-26T00:00:01Z",
        "model": "stt-rt-v3",
        "filename": "test.mp3",
        "enable_speaker_diarization": False,
        "enable_language_identification": False,
        "audio_duration_ms": 0,
        "error_message": "Invalid audio format",
    }

    mock_client.post.side_effect = [mock_upload_resp, mock_create_resp]
    mock_client.get.return_value = mock_status_resp
    mock_client.delete.return_value = MagicMock(is_error=False)

    loader = SonioxDocumentLoader(file_path="test.mp3", api_key="test_key")

    with pytest.raises(SonioxTranscriptionFailedError, match="Invalid audio format"):
        async for _ in loader.alazy_load():
            pass

    # Verify cleanup
    assert mock_client.delete.call_count == 2
