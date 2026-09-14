import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import _resolve_key, set_session_api_key  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Each test starts with no ambient key and no session override."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    set_session_api_key(None)
    yield
    set_session_api_key(None)


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

def test_session_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_owner_shared")
    set_session_api_key("gsk_visitor_private")
    assert _resolve_key("GROQ_API_KEY") == "gsk_visitor_private"


def test_falls_back_to_env_when_no_session_key(monkeypatch):
    """A visitor who pastes nothing uses the deployer's key."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_owner_shared")
    assert _resolve_key("GROQ_API_KEY") == "gsk_owner_shared"


def test_blank_and_whitespace_keys_are_ignored(monkeypatch):
    """An empty text_input must not shadow the owner key with ''."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_owner_shared")
    for blank in ("", "   ", "\n", None):
        set_session_api_key(blank)
        assert _resolve_key("GROQ_API_KEY") == "gsk_owner_shared", f"blank {blank!r} leaked through"


def test_returns_none_when_nothing_configured():
    assert _resolve_key("GROQ_API_KEY") is None


def test_never_writes_to_os_environ():
    """The whole point: setting a session key must leave os.environ alone."""
    set_session_api_key("gsk_visitor_private")
    assert os.getenv("GROQ_API_KEY") is None, "session key leaked into os.environ"


# ---------------------------------------------------------------------------
# The actual leak: two concurrent visitors
# ---------------------------------------------------------------------------

def test_concurrent_sessions_do_not_see_each_others_keys(monkeypatch):
    """Reproduces the exact interleaving that leaked before the fix.

    Visitor B opens the page, visitor A then pastes her private key, and B
    clicks Run afterwards. With the os.environ implementation B's run used
    'gsk_A_private'. Each thread is its own ContextVar context, so now it
    doesn't."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_owner_shared")
    seen = {}

    def session(name, typed, wait_before_set, wait_before_run):
        time.sleep(wait_before_set)
        set_session_api_key(typed or os.getenv("GROQ_API_KEY"))
        time.sleep(wait_before_run)
        seen[name] = _resolve_key("GROQ_API_KEY")

    b = threading.Thread(target=session, args=("B", "", 0.00, 0.40))
    a = threading.Thread(target=session, args=("A", "gsk_A_private", 0.10, 0.05))
    b.start(); a.start(); b.join(); a.join()

    assert seen["A"] == "gsk_A_private", "visitor A's own key was ignored"
    assert seen["B"] == "gsk_owner_shared", f"visitor B leaked A's key: {seen['B']}"


def test_many_concurrent_sessions_stay_isolated():
    """Ten visitors, ten different keys, interleaved."""
    seen = {}

    def session(i):
        set_session_api_key(f"gsk_key_{i}")
        time.sleep(0.02 * (i % 5))          # force interleaving
        seen[i] = _resolve_key("GROQ_API_KEY")

    threads = [threading.Thread(target=session, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(10):
        assert seen[i] == f"gsk_key_{i}", f"session {i} saw {seen[i]}"


def test_cli_path_unaffected(monkeypatch):
    """`python -m src.graph` never calls set_session_api_key, so it must keep
    reading straight from the environment / .env."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_dotenv")
    assert _resolve_key("GROQ_API_KEY") == "gsk_from_dotenv"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))