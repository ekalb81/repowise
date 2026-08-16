"""The server must not die because an embedder key is missing.

``uvicorn repowise.server.app:create_app`` is what docker/entrypoint.sh runs,
and ``_build_embedder`` used to read only ``os.environ``. A configured backend
whose key lives in a config file therefore raised inside the lifespan, which
aborts startup — so the container crash-loops on a traceback and serves no
wiki, no graph and no health page either.

The full-text index above it already made this call for the same reason
(issue #1309): degrade, keep serving, and say so.
"""

from __future__ import annotations

import os

import pytest

from repowise.core.providers.embedding.base import KeylessEmbedder
from repowise.server import app as app_module


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """Isolate from the developer's own ~/.repowise/config.yaml.

    The key resolver reads the global config, so on a machine that has a real
    embedder key saved there these tests would build a working embedder and
    the degradation cases could never fire. The env-var isolation in
    tests/conftest.py does not cover a file.
    """
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)
        app_module._embedder_status.update({"name": "mock", "degraded": False, "reason": None})


def _no_persisted_key(monkeypatch):
    monkeypatch.setattr(app_module, "_embedder_api_key", lambda name: None)


# ---------------------------------------------------------------------------
# Degrade rather than abort
# ---------------------------------------------------------------------------


def test_missing_key_degrades_instead_of_raising(monkeypatch):
    """The regression: this used to raise and take startup down with it."""
    _no_persisted_key(monkeypatch)
    monkeypatch.setenv("REPOWISE_EMBEDDER", "voxell")
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)

    embedder = app_module._build_embedder()

    assert isinstance(embedder, KeylessEmbedder)
    assert app_module._embedder_status["degraded"] is True
    assert app_module._embedder_status["name"] == "voxell"


def test_the_reason_names_the_failure(monkeypatch):
    _no_persisted_key(monkeypatch)
    monkeypatch.setenv("REPOWISE_EMBEDDER", "voxell")
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)

    app_module._build_embedder()

    assert "API key" in (app_module._embedder_status["reason"] or "")


def test_mock_default_is_not_a_degradation(monkeypatch):
    monkeypatch.delenv("REPOWISE_EMBEDDER", raising=False)

    embedder = app_module._build_embedder()

    assert isinstance(embedder, KeylessEmbedder)
    assert app_module._embedder_status["degraded"] is False


# ---------------------------------------------------------------------------
# The key is resolved from where it is actually persisted
# ---------------------------------------------------------------------------


def test_persisted_key_is_used_when_the_env_has_none(monkeypatch):
    """What docker/uvicorn could not do before: read a key it was not handed."""
    monkeypatch.setenv("REPOWISE_EMBEDDER", "voxell")
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "_embedder_api_key", lambda name: "vf_from_config")

    embedder = app_module._build_embedder()

    assert type(embedder).__name__ == "VoxellEmbedder"
    assert app_module._embedder_status["degraded"] is False


def test_exported_key_still_works(monkeypatch):
    _no_persisted_key(monkeypatch)
    monkeypatch.setenv("REPOWISE_EMBEDDER", "voxell")
    monkeypatch.setenv("VOXELL_API_KEY", "vf_exported")

    embedder = app_module._build_embedder()

    assert type(embedder).__name__ == "VoxellEmbedder"
    assert app_module._embedder_status["degraded"] is False


def test_key_lookup_failure_is_not_fatal(monkeypatch):
    """The resolver is a diagnostic path; it must not become the crash.

    Patches the real import target — the lookup is imported inside the
    function from the MCP server module, so patching a name on ``app`` would
    silently do nothing and the test would pass for the wrong reason.
    """

    def _boom(name):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(
        "repowise.server.mcp_server._server._persisted_embedder_key", _boom
    )

    assert app_module._embedder_api_key("voxell") is None  # swallowed, not raised

    monkeypatch.setenv("REPOWISE_EMBEDDER", "voxell")
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)
    assert isinstance(app_module._build_embedder(), KeylessEmbedder)


# ---------------------------------------------------------------------------
# /api/health reports it
# ---------------------------------------------------------------------------


def test_health_reports_a_degraded_embedder():
    from repowise.server.schemas.health import HealthResponse

    resp = HealthResponse(
        status="degraded", db="ok", version="0",
        embedder="voxell", embedder_degraded=True, embedder_reason="Voxell API key required",
    )
    assert resp.embedder_degraded is True
    assert resp.embedder == "voxell"


def test_health_fields_default_to_absent():
    """Older callers constructing HealthResponse without them keep working."""
    from repowise.server.schemas.health import HealthResponse

    resp = HealthResponse(status="healthy", db="ok", version="0")
    assert resp.embedder is None
    assert resp.embedder_degraded is False
