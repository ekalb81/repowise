"""Ambient environment must not steer the code under test, or the real thing.

Two halves. The first is about the suite: resolution is environment-first by
design, so a developer's exported key changes what the code resolves and the
tests asserting a *default* fail on their machine while passing in CI. The
second is about the product: the same precedence silently discards what a repo
saved, and nothing used to say so.
"""

from __future__ import annotations

import os

from repowise.core.repo_config import env_conflicts, save_repo_env_key

# ---------------------------------------------------------------------------
# The suite is isolated from the developer's environment
# ---------------------------------------------------------------------------

_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "VOXELL_API_KEY",
    "DEEPSEEK_API_KEY",
    "KIMI_API_KEY",
)


def test_no_provider_credential_is_visible_to_tests():
    """The fixture ran. A key here means a machine-dependent suite."""
    leaked = [name for name in _CREDENTIAL_VARS if os.environ.get(name)]
    assert leaked == [], (
        f"{leaked} visible to tests — resolution is environment-first, so these "
        "change what the code under test resolves. Add them to the isolation "
        "fixture in tests/conftest.py."
    )


def test_nothing_the_fixture_stripped_came_back(stripped_ambient_vars):
    """Scoped to what this machine actually had, not to every ``REPOWISE_*``.

    A blanket "no REPOWISE_ variable exists" assertion fails for a correct
    reason: another session fixture sets ``REPOWISE_SKIP_EDITOR_SETUP`` on
    purpose so tests cannot write real editor config. What must hold is that
    the developer's own exported values stay hidden.
    """
    returned = sorted(name for name in stripped_ambient_vars if name in os.environ)
    assert returned == []


def test_isolation_covers_every_llm_registry_credential(ambient_steering_vars):
    """Drift guard: a provider added to the registry is isolated automatically.

    The set is derived from these tables rather than retyped, so this asserts
    the derivation still reaches all of them.
    """
    from repowise.core.providers.llm.registry import (
        PROVIDER_API_KEY_ENVS,
        PROVIDER_BASE_URL_ENVS,
    )

    for table in (PROVIDER_API_KEY_ENVS, PROVIDER_BASE_URL_ENVS):
        for provider, group in table.items():
            for name in group:
                assert name in ambient_steering_vars, f"{name} ({provider}) is not isolated"


def test_isolation_covers_every_embedder_credential(ambient_steering_vars):
    """The embedder table is the one that does not overlap the LLM providers.

    ``voxell`` has no LLM counterpart, so nothing else would have covered
    ``VOXELL_API_KEY``.
    """
    from repowise.server.mcp_server._server import _EMBEDDER_KEY_ENV

    for embedder, group in _EMBEDDER_KEY_ENV.items():
        for name in group:
            assert name in ambient_steering_vars, f"{name} ({embedder}) is not isolated"


# ---------------------------------------------------------------------------
# The product says when the shell overrode what the repo saved
# ---------------------------------------------------------------------------


def _repo_with_saved_key(tmp_path, name: str = "VOXELL_API_KEY", value: str = "vf_saved"):
    (tmp_path / ".repowise").mkdir(parents=True, exist_ok=True)
    save_repo_env_key(tmp_path, name, value)
    return tmp_path


def test_no_conflict_when_the_environment_is_silent(tmp_path):
    repo = _repo_with_saved_key(tmp_path)
    assert env_conflicts(repo, environ={}) == []


def test_no_conflict_when_both_sides_agree(tmp_path):
    repo = _repo_with_saved_key(tmp_path)
    assert env_conflicts(repo, environ={"VOXELL_API_KEY": "vf_saved"}) == []


def test_conflict_when_the_environment_holds_a_different_value(tmp_path):
    repo = _repo_with_saved_key(tmp_path)
    assert env_conflicts(repo, environ={"VOXELL_API_KEY": "vf_exported"}) == ["VOXELL_API_KEY"]


def test_conflicts_are_names_only_never_values(tmp_path):
    """These are credentials; a diagnostic must not print either side."""
    repo = _repo_with_saved_key(tmp_path, "OPENAI_API_KEY", "sk-saved-secret")
    result = env_conflicts(repo, environ={"OPENAI_API_KEY": "sk-exported-secret"})
    assert result == ["OPENAI_API_KEY"]
    joined = " ".join(result)
    assert "sk-saved-secret" not in joined
    assert "sk-exported-secret" not in joined


def test_missing_env_file_is_not_a_conflict(tmp_path):
    assert env_conflicts(tmp_path, environ={"OPENAI_API_KEY": "sk-x"}) == []


def test_load_dotenv_warns_about_the_shadowed_name(tmp_path, monkeypatch, capsys):
    """The regression this exists for: an exported key silently beating the
    saved one, and the 401 that follows naming neither."""
    from repowise.cli.ui import load_dotenv

    repo = _repo_with_saved_key(tmp_path)
    monkeypatch.setenv("VOXELL_API_KEY", "vf_exported")
    load_dotenv(repo)

    err = capsys.readouterr().err
    assert "VOXELL_API_KEY" in err
    assert ".repowise/.env" in err
    # The exported value still wins — the warning reports, it does not change it.
    assert os.environ["VOXELL_API_KEY"] == "vf_exported"
    # And it never prints either secret.
    assert "vf_saved" not in err
    assert "vf_exported" not in err


def test_load_dotenv_is_quiet_when_nothing_is_contested(tmp_path, monkeypatch, capsys):
    from repowise.cli.ui import load_dotenv

    repo = _repo_with_saved_key(tmp_path)
    monkeypatch.delenv("VOXELL_API_KEY", raising=False)
    load_dotenv(repo)

    assert "overrides" not in capsys.readouterr().err
    assert os.environ["VOXELL_API_KEY"] == "vf_saved"
