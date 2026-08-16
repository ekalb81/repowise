"""``serve`` must not start an embedder it has no credential for.

The repo-config pass sets ``REPOWISE_EMBEDDER`` from ``config.yaml``, and
``_setup_embedder`` used to treat "already set" as "nothing to do" — so the key
restore never ran and the app built a keyless embedder, which aborts startup:

    ValueError: Voxell API key required. Pass api_key= or set VOXELL_API_KEY

It stayed hidden while the keyed embedders were ones whose variable tends to be
exported anyway. An embedder whose key lives only in a config file fails every
time.
"""

from __future__ import annotations

import os

import pytest
import yaml

from repowise.cli.commands.serve_cmd import _ensure_embedder_key, _setup_embedder


@pytest.fixture(autouse=True)
def _restore_environ():
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def _global_config(tmp_path, monkeypatch, **values):
    """Point Path.home() at a tmp dir holding ~/.repowise/config.yaml."""
    cfg_dir = tmp_path / ".repowise"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(yaml.dump(values), encoding="utf-8")
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_pinned_embedder_still_gets_its_key(tmp_path, monkeypatch):
    """The regression: env names the embedder, config holds the key."""
    _global_config(tmp_path, monkeypatch, embedder="voxell", embedder_api_key="vf_from_config")
    monkeypatch.setenv("REPOWISE_EMBEDDER", "voxell")
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)

    _setup_embedder()

    assert os.environ.get("VOXELL_API_KEY") == "vf_from_config"


def test_exported_key_is_never_overwritten(tmp_path, monkeypatch):
    _global_config(tmp_path, monkeypatch, embedder="voxell", embedder_api_key="vf_from_config")
    monkeypatch.setenv("REPOWISE_EMBEDDER", "voxell")
    monkeypatch.setenv("VOXELL_API_KEY", "vf_exported")

    _setup_embedder()

    assert os.environ["VOXELL_API_KEY"] == "vf_exported"


def test_key_comes_from_the_repo_env_too(tmp_path, monkeypatch):
    """A key saved per-repo counts, not only the global config."""
    from repowise.core.repo_config import save_repo_env_key

    repo = tmp_path / "repo"
    (repo / ".repowise").mkdir(parents=True)
    save_repo_env_key(repo, "VOXELL_API_KEY", "vf_from_repo")
    _global_config(tmp_path, monkeypatch, embedder="mock")
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)

    _ensure_embedder_key("voxell", repo)

    assert os.environ.get("VOXELL_API_KEY") == "vf_from_repo"


def test_keyless_embedder_needs_nothing(tmp_path, monkeypatch):
    """ollama and mock have no credential; resolution must not invent one."""
    _global_config(tmp_path, monkeypatch, embedder="voxell", embedder_api_key="vf_from_config")
    monkeypatch.setenv("REPOWISE_EMBEDDER", "ollama")

    _setup_embedder()

    assert "VOXELL_API_KEY" not in os.environ


def test_no_key_anywhere_is_not_a_crash(tmp_path, monkeypatch):
    """Absent stays absent — the embedder raises its own message later."""
    _global_config(tmp_path, monkeypatch, embedder="mock")
    monkeypatch.setenv("REPOWISE_EMBEDDER", "voxell")
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)

    _setup_embedder()

    assert "VOXELL_API_KEY" not in os.environ
