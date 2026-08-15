"""Unit tests for VoxellEmbedder.

All tests mock openai.OpenAI — no real API calls are made.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai", reason="openai SDK not installed")

from repowise.core.providers.embedding.voxell import VoxellEmbedder

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Voxell API key required"):
        VoxellEmbedder(api_key=None)


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("VOXELL_API_KEY", "vf_test")
    emb = VoxellEmbedder()
    assert emb._api_key == "vf_test"


def test_openai_key_does_not_satisfy_voxell(monkeypatch):
    """The whole reason this embedder exists: an unrelated OpenAI key must not
    be picked up, because sending it to Voxell yields an opaque 401."""
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-unrelated")
    with pytest.raises(ValueError, match="Voxell API key required"):
        VoxellEmbedder()


def test_default_model():
    emb = VoxellEmbedder(api_key="k")
    assert emb._model == "forge-turbo"
    assert emb.dimensions == 1024


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("forge-turbo", 1024),
        ("forge-pro", 2560),
        ("forge-ultra-4k", 4096),
        ("text-embedding-3-small", 1536),
        ("text-embedding-3-large", 3072),
    ],
)
def test_known_model_dimensions(model, expected):
    assert VoxellEmbedder(api_key="k", model=model).dimensions == expected


def test_unknown_model_raises_at_construction():
    """Unknown models must fail fast — a silent dim fallback would corrupt the vector store."""
    with pytest.raises(ValueError, match="Unknown Voxell embedding model"):
        VoxellEmbedder(api_key="k", model="forge-future")


def test_unknown_model_accepted_with_explicit_dims():
    emb = VoxellEmbedder(api_key="k", model="forge-future", dimensions=512)
    assert emb.dimensions == 512


def test_dims_from_env(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "256")
    assert VoxellEmbedder(api_key="k").dimensions == 256


def test_malformed_env_raises_the_same_message(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "abc")
    with pytest.raises(ValueError, match="dimensions must be a positive integer"):
        VoxellEmbedder(api_key="k")


def test_base_url_override_from_env(monkeypatch):
    monkeypatch.setenv("VOXELL_BASE_URL", "http://localhost:9999/v1")
    assert VoxellEmbedder(api_key="k")._base_url == "http://localhost:9999/v1"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _make_mock_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = []
    for values in vectors:
        item = MagicMock()
        item.embedding = values
        response.data.append(item)
    return response


async def test_embed_empty_returns_empty():
    emb = VoxellEmbedder(api_key="k")
    assert await emb.embed([]) == []


async def test_embed_returns_normalized_vectors():
    emb = VoxellEmbedder(api_key="k", dimensions=3)

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.return_value = _make_mock_response(
            [[3.0, 0.0, 4.0]]
        )
        result = await emb.embed(["hello"])

    assert len(result) == 1
    assert abs(math.sqrt(sum(x * x for x in result[0])) - 1.0) < 1e-6


async def test_embed_batch_returns_correct_count():
    emb = VoxellEmbedder(api_key="k", dimensions=2)

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.return_value = _make_mock_response(
            [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
        )
        result = await emb.embed(["a", "b", "c"])

    assert len(result) == 3


async def test_embed_uses_voxell_base_url():
    emb = VoxellEmbedder(api_key="vf_test", dimensions=1)

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.return_value = _make_mock_response([[1.0]])
        await emb.embed(["test"])

    assert mock_client.call_args.kwargs.get("base_url") == "https://api.voxell.ai/v1"


async def test_dimensions_sent_only_when_overridden():
    """A stock request stays byte-identical; an override is sent so the returned
    vectors match the declared width."""
    captured: list[dict] = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        return _make_mock_response([[1.0] * 1024])

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.side_effect = fake_create
        await VoxellEmbedder(api_key="k").embed(["stock"])
    assert "dimensions" not in captured[0]
    assert captured[0]["model"] == "forge-turbo"
    assert captured[0]["input"] == ["stock"]

    captured.clear()

    def fake_create_512(**kwargs):
        captured.append(kwargs)
        return _make_mock_response([[1.0] * 512])

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.side_effect = fake_create_512
        await VoxellEmbedder(api_key="k", dimensions=512).embed(["override"])
    assert captured[0]["dimensions"] == 512


# ---------------------------------------------------------------------------
# Width verification
# ---------------------------------------------------------------------------


async def test_embed_raises_when_api_returns_wrong_width():
    emb = VoxellEmbedder(api_key="k", model="forge-turbo")
    assert emb.dimensions == 1024

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.return_value = _make_mock_response(
            [[1.0, 0.0, 0.0]]
        )
        with pytest.raises(ValueError, match="1024") as exc_info:
            await emb.embed(["hello"])

    msg = str(exc_info.value)
    assert "3" in msg
    assert "_DIMS" in msg


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registered_in_embedder_registry():
    from repowise.core.providers.embedding.registry import get_embedder, list_embedders

    assert "voxell" in list_embedders()
    assert isinstance(get_embedder("voxell", api_key="k"), VoxellEmbedder)


def test_mcp_server_knows_the_key_env():
    """Without this entry the MCP server cannot recover a persisted Voxell key
    and silently degrades to mock vectors."""
    from repowise.server.mcp_server._server import _EMBEDDER_KEY_ENV, _EMBEDDER_REMEDIATION

    assert _EMBEDDER_KEY_ENV["voxell"] == ("VOXELL_API_KEY",)
    assert "VOXELL_API_KEY" in _EMBEDDER_REMEDIATION["voxell"]
