"""Root test configuration — fixtures available to all test modules."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Steering variables that live outside the provider/embedder registry tables:
# either they belong to an embedder with no LLM counterpart (voxell), or they
# tune a backend without naming a credential (the ollama trio), or they point
# persistence somewhere else entirely (DATABASE_URL).
_AMBIENT_EXTRA_VARS = frozenset(
    {
        "VOXELL_API_KEY",
        "VOXELL_BASE_URL",
        "OLLAMA_EMBEDDING_MODEL",
        "OLLAMA_EMBEDDING_DIMS",
        "OLLAMA_EMBEDDING_TIMEOUT",
        "DATABASE_URL",
    }
)


def _ambient_steering_vars() -> set[str]:
    """Every environment variable that can steer provider/embedder resolution.

    Derived from the registry tables rather than retyped, so a provider added
    there is isolated here without anyone remembering. ``REPOWISE_*`` is taken
    by prefix for the same reason: the config knobs (``REPOWISE_PROVIDER``,
    ``REPOWISE_MODEL``, ``REPOWISE_EMBEDDER``, the embedding trio, reasoning,
    the database URLs) all share it, and a new one should not need a second
    edit here to be covered.
    """
    names: set[str] = set(_AMBIENT_EXTRA_VARS)
    try:
        from repowise.core.providers.llm.registry import (
            PROVIDER_API_KEY_ENVS,
            PROVIDER_BASE_URL_ENVS,
        )
    except Exception:  # pragma: no cover - registry import is not test-critical
        pass
    else:
        for table in (PROVIDER_API_KEY_ENVS, PROVIDER_BASE_URL_ENVS):
            for group in table.values():
                names.update(group)
    names.update(name for name in os.environ if name.startswith("REPOWISE_"))
    return names


@pytest.fixture(scope="session", autouse=True)
def _isolate_home(tmp_path_factory: pytest.TempPathFactory):
    """Point the home directory at an empty temp dir for the whole session.

    Stripping ambient *variables* is only half of it: a good deal of what
    repowise resolves lives in ambient *files* under the home directory, and
    the env fixture below cannot see those. The global
    ``~/.repowise/config.yaml`` holds an ``embedder`` and its
    ``embedder_api_key``, which is enough to make a test that should observe a
    missing credential quietly observe a working one — that is how a
    degradation test on this machine built a real embedder and passed for the
    wrong reason. The agent-wiring code probes ``~/.claude``, ``~/.codex`` and
    ``~/.cursor`` the same way, so "is this agent installed" answers
    differently on a developer's box than in CI.

    Only ``Path.home()`` is redirected, deliberately, and not the environment
    that ``os.path.expanduser("~")`` reads. Redirecting both looked tidier and
    broke a real guard: ``rewrite_hook._find_repo_root`` walks up from cwd and
    refuses to treat *the home directory* as a repo root, and on Windows the
    temp directory lives under the user profile — so every ``tmp_path`` walk-up
    reaches the developer's real home. With that home no longer recognised as
    home, a stray ``~/.repowise`` there captured tmp dirs that must pass
    through untouched.

    So the split is the point: the config readers this fixture exists for go
    through ``Path.home()``, while the guards that need to recognise the
    machine's actual home keep reading it.
    """
    home = tmp_path_factory.mktemp("home")
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    mp.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    try:
        yield home
    finally:
        mp.undo()


@pytest.fixture(scope="session", autouse=True)
def _isolate_ambient_env():
    """Hide the developer's own credentials and config from every test.

    Resolution is deliberately environment-first — ``REPOWISE_PROVIDER`` beats
    ``config.yaml``, an exported key beats ``.repowise/.env`` — so a machine
    that exports one silently changes what the code under test resolves. The
    tests that assert a *default* are the ones that break: a developer with
    ``OPENAI_API_KEY`` set watches ``test_advanced_config_default_keys_no_fast``
    read ``embedder: openai`` where it expects ``mock``, and
    ``test_update_provider_prompt`` see no prompt because credentials were
    already found. Both pass in CI and fail on the machine, which reads as
    flakiness rather than as a leak.

    Session scope, not function: this is about the ambient environment the
    process started with. Tests that want a variable set continue to use
    ``monkeypatch.setenv``, which runs after this and is undone per test.
    """
    saved = {}
    for name in _ambient_steering_vars():
        if name in os.environ:
            saved[name] = os.environ.pop(name)
    try:
        yield saved
    finally:
        os.environ.update(saved)


@pytest.fixture(scope="session")
def ambient_steering_vars() -> set[str]:
    """The isolated variable set, exposed so the drift test can check coverage."""
    return _ambient_steering_vars()


@pytest.fixture(scope="session")
def stripped_ambient_vars(_isolate_ambient_env) -> dict[str, str]:
    """What the isolation fixture actually removed from this machine.

    The thing worth asserting is that none of *these* came back, not that no
    ``REPOWISE_*`` variable exists at all: other session fixtures set some on
    purpose (``REPOWISE_SKIP_EDITOR_SETUP`` keeps tests from writing real
    editor config), and a blanket assertion turns their correct behaviour into
    a failure.
    """
    return _isolate_ambient_env


@pytest.fixture(scope="session", autouse=True)
def _no_telemetry_network(tmp_path_factory: pytest.TempPathFactory):
    """Guarantee no test emits telemetry over the network or into real state.

    The MCP instrument seam emits an ``mcp_tool_call`` event via the core
    emitter's ``_post``; a test that drives the real wrapper with consent
    enabled would otherwise POST to the production ingest endpoint. Patch that
    sink to a no-op. Tests that assert emit behaviour re-patch it at function
    scope and still never touch the network.

    The CLI's ``command_run`` path is not patched here — its tests patch
    ``PlatformClient.post`` at the class level, so patching the
    ``default_client`` instance would shadow those patches. Its delivery is
    already off under pytest (``emitter._under_test``); what this fixture adds
    is redirecting the event spool, so a test that records an event cannot
    leave it queued in the real ``~/.repowise`` for a later real invocation to
    deliver.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        from repowise.core.platform import telemetry as _core_telemetry

        mp.setattr(_core_telemetry, "_post", lambda envelope: None, raising=False)
    except Exception:
        pass
    try:
        from repowise.cli.platform.telemetry import spool as _cli_spool

        spool_path = tmp_path_factory.mktemp("telemetry") / _cli_spool.SPOOL_FILENAME
        mp.setattr(_cli_spool, "_path", lambda: spool_path)
    except Exception:
        pass
    yield
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def _no_real_editor_setup():
    """Guarantee no test repoints the developer's real global editor config.

    ``repowise init`` (and ``doctor``'s self-heal path) defaults to
    ``--editor-setup`` on, which rewrites the *global*, machine-wide MCP
    server registration (e.g. ``~/.claude/settings.json``) to point at
    whatever repo path was just indexed. A test that drives the real CLI
    against a ``tmp_path`` fixture repo — e.g.
    ``tests/integration/test_cli.py``'s lock/watcher tests, which call
    ``init`` through ``CliRunner`` in-process rather than mocking it out —
    was doing exactly that: it repointed the developer's actual Claude Code
    MCP entry at a pytest temp directory that gets wiped on reboot, breaking
    every other project's ``repowise`` MCP tools until the *next*
    ``doctor``/``update`` run happened to self-heal it back (and even then,
    an already-running Claude Code session keeps using the stale spawn
    command it cached at connect time, since it has no reason to re-read
    the file mid-session).

    ``REPOWISE_SKIP_EDITOR_SETUP`` is the exact env var editor_setup.py and
    doctor's self-heal migrations already gate on — set once, for the whole
    session, so no test (present or future) can hit this by omission the
    way the lock/watcher tests did.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    mp.setenv("REPOWISE_SKIP_EDITOR_SETUP", "1")
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _isolate_structlog_config():
    """Restore structlog's global configuration after every test.

    ``configure_cli_logging`` / ``silence_logs_for_machine_output`` install a
    filtering bound logger at ERROR process-wide and never undo it, so the
    first test to exercise a CLI command silences every ``info`` and
    ``warning`` for the rest of the session.

    That is invisible until a test asserts on a log record.
    ``structlog.testing.capture_logs`` swaps the *processor chain*, not the
    *wrapper class*, so a filtering logger drops the event before ``LogCapture``
    ever runs and the test reads an empty list. Tests collected after
    ``tests/unit/cli`` therefore pass alone and fail in a full run.

    Snapshot and restore rather than reset to defaults: a test that configures
    structlog on purpose keeps working, and the next test still starts clean.
    """
    import structlog

    saved = structlog.get_config()
    try:
        yield
    finally:
        structlog.configure(**saved)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_repo_path(repo_root: Path) -> Path:
    """Path to the multi-language sample repository used in integration tests."""
    path = repo_root / "tests" / "fixtures" / "sample_repo"
    assert path.exists(), (
        f"Sample repo not found at {path}. Run 'make install' to ensure test fixtures are in place."
    )
    return path


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    """Path to the tests/fixtures/ directory."""
    return repo_root / "tests" / "fixtures"
