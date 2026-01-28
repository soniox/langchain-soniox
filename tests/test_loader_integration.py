"""
Integration tests for SonioxDocumentLoader.

These tests make real API calls to Soniox and require a valid API key.
Set the SONIOX_API_KEY environment variable to run these tests.

If SONIOX_API_KEY is not set, all integration tests will be skipped.
"""

import os

import httpx
import pytest

from langchain_soniox import (
    SonioxDocumentLoader,
    SonioxTranscriptionOptions,
    StructuredContext,
    StructuredContextGeneralItem,
    TranslationConfig,
    StructuredContextTranslationTerm,
)

# Skip all tests in this file if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("SONIOX_API_KEY"),
    reason="SONIOX_API_KEY environment variable not set",
)

# Example audio file URL from Soniox
EXAMPLE_AUDIO_URL = "https://soniox.com/media/examples/coffee_shop.mp3"


@pytest.fixture
def example_audio_file(tmp_path):
    """Download the example audio file for testing."""
    audio_path = tmp_path / "coffee_shop.mp3"

    # Download the file
    response = httpx.get(EXAMPLE_AUDIO_URL)
    response.raise_for_status()

    audio_path.write_bytes(response.content)
    return audio_path


@pytest.fixture
def example_audio_bytes():
    """Get the example audio file as bytes."""
    response = httpx.get(EXAMPLE_AUDIO_URL)
    response.raise_for_status()
    return response.content


# Sync tests


def test_sync_load_with_file_url():
    """Test sync loading with a remote file URL."""
    print("Testing sync load with file_url")

    loader = SonioxDocumentLoader(file_url=EXAMPLE_AUDIO_URL)
    docs = list(loader.lazy_load())

    assert len(docs) == 1
    doc = docs[0]

    print("Transcription successful")
    print(f"Text: {doc.page_content[:100]}...")
    print(f"Duration: {doc.metadata['audio_duration_ms']}ms")
    print(f"Tokens: {len(doc.metadata['tokens'])} tokens")

    # Basic assertions
    assert doc.page_content
    assert doc.metadata["source"] == EXAMPLE_AUDIO_URL
    assert doc.metadata["transcription_id"]
    assert doc.metadata["audio_duration_ms"] > 0
    assert len(doc.metadata["tokens"]) > 0


def test_sync_load_with_file_path(example_audio_file):
    """Test sync loading with a local file path."""
    print("Testing sync load with file_path")

    loader = SonioxDocumentLoader(file_path=str(example_audio_file))
    docs = list(loader.lazy_load())

    assert len(docs) == 1
    doc = docs[0]

    print("Transcription successful")
    print(f"Text: {doc.page_content[:100]}...")
    print(f"Duration: {doc.metadata['audio_duration_ms']}ms")
    print(f"Tokens: {len(doc.metadata['tokens'])} tokens")

    assert doc.page_content
    assert doc.metadata["source"] == str(example_audio_file)
    assert doc.metadata["transcription_id"]


def test_sync_load_with_file_data(example_audio_bytes):
    """Test sync loading with binary file data."""
    print("Testing sync load with file_data")

    loader = SonioxDocumentLoader(file_data=example_audio_bytes)
    docs = list(loader.lazy_load())

    assert len(docs) == 1
    doc = docs[0]

    print("Transcription successful")
    print(f"Text: {doc.page_content[:100]}...")
    print(f"Duration: {doc.metadata['audio_duration_ms']}ms")
    print(f"Tokens: {len(doc.metadata['tokens'])} tokens")

    assert doc.page_content
    assert doc.metadata["source"] == "file_upload"
    assert doc.metadata["transcription_id"]


# Async Tests


async def test_async_load_with_file_url():
    """Test async loading with a remote file URL."""
    print("Testing async load with file_url")

    loader = SonioxDocumentLoader(file_url=EXAMPLE_AUDIO_URL)
    docs = [doc async for doc in loader.alazy_load()]

    assert len(docs) == 1
    doc = docs[0]

    print("Async transcription successful")
    print(f"Text: {doc.page_content[:100]}...")
    print(f"Duration: {doc.metadata['audio_duration_ms']}ms")
    print(f"Tokens: {len(doc.metadata['tokens'])} tokens")

    assert doc.page_content
    assert doc.metadata["source"] == EXAMPLE_AUDIO_URL
    assert doc.metadata["transcription_id"]


async def test_async_load_with_file_path(example_audio_file):
    """Test async loading with a local file path."""
    print("Testing async load with file_path")

    loader = SonioxDocumentLoader(file_path=str(example_audio_file))
    docs = [doc async for doc in loader.alazy_load()]

    assert len(docs) == 1
    doc = docs[0]

    print("Async transcription successful")
    print(f"Text: {doc.page_content[:100]}...")

    assert doc.page_content
    assert doc.metadata["source"] == str(example_audio_file)


async def test_async_load_with_file_data(example_audio_bytes):
    """Test async loading with binary file data."""
    print("Testing async load with file_data")

    loader = SonioxDocumentLoader(file_data=example_audio_bytes)
    docs = [doc async for doc in loader.alazy_load()]

    assert len(docs) == 1
    doc = docs[0]

    print("Async transcription successful")
    print(f"Text: {doc.page_content[:100]}...")

    assert doc.page_content
    assert doc.metadata["source"] == "file_upload"


# Test custom model options


def test_with_custom_options():
    print("Testing with custom transcription options")

    options = SonioxTranscriptionOptions(
        language_hints=["en"],
        enable_speaker_diarization=True,
        enable_language_identification=True,
        context=StructuredContext(
            general=[
                StructuredContextGeneralItem(
                    key="Topic",
                    value="Coffee shop conversation",
                )
            ],
            terms=["cappucino", "espresso", "latte"],
            text="This is a recording of a conversation in a coffee shop.",
        ),
    )

    loader = SonioxDocumentLoader(
        file_url=EXAMPLE_AUDIO_URL,
        options=options,
    )

    docs = list(loader.lazy_load())
    assert len(docs) == 1

    doc = docs[0]
    print("Custom options applied successfully")

    assert doc.metadata["tokens"][0]["speaker"] is not None
    assert doc.metadata["tokens"][0]["language"] is not None


def test_with_translation_config():
    print("Testing with custom transcription options: translation")

    options = SonioxTranscriptionOptions(
        enable_language_identification=True,
        translation=TranslationConfig(
            type="one_way",
            target_language="fr",
        ),
        context=StructuredContext(
            translation_terms=[
                StructuredContextTranslationTerm(
                    source="latte",
                    target="cafe au lait",
                )
            ]
        ),
    )

    loader = SonioxDocumentLoader(
        file_url=EXAMPLE_AUDIO_URL,
        options=options,
    )

    docs = list(loader.lazy_load())
    assert len(docs) == 1

    doc = docs[0]
    print("Custom options applied successfully")

    assert doc.metadata["tokens"][0]["language"] is not None
    assert doc.metadata["tokens"][0]["translation_status"] == "original"
