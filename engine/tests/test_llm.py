"""Tests for the LLM provider layer, against a fake HTTP server (no network, no real keys)."""

import json
import logging
import pickle

import httpx
import pytest

from eda_engine.llm import (
    DEFAULT_CHAIN,
    KEY_ENV_VARS,
    AllProvidersFailed,
    ApiKey,
    BudgetExceeded,
    KeyRing,
    LLMClient,
    LLMOutputError,
    TokenBudget,
    client_from_env,
    estimate_messages,
    parse_json,
    redact,
)

GROQ_KEY, CEREBRAS_KEY, OPENROUTER_KEY = "gsk_test_groq_secret_1", "csk_test_cerebras_2", "sk-or-test-router-3"
HOSTS = {"api.groq.com": "groq", "api.cerebras.ai": "cerebras", "openrouter.ai": "openrouter"}
MESSAGES = [{"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": "Summarise: credit_score drives default."}]


def ok(text="All good.", usage=(120, 30), model=None):
    body = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
    if model:
        body["model"] = model
    return httpx.Response(200, json=body)


class FakeServers:
    """Scripted responses per provider; records every request."""

    def __init__(self, **scripts):
        self.scripts = {name: list(responses) for name, responses in scripts.items()}
        self.requests: list[tuple[str, httpx.Request]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        provider = HOSTS[request.url.host]
        self.requests.append((provider, request))
        script = self.scripts.get(provider) or [ok()]
        item = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(item, Exception):
            raise item
        return item

    def called(self) -> list[str]:
        return [p for p, _ in self.requests]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def all_keys():
    return KeyRing(server={"groq": ApiKey(GROQ_KEY), "cerebras": ApiKey(CEREBRAS_KEY),
                           "openrouter": ApiKey(OPENROUTER_KEY)})


def make_client(servers, keys=None, budget=None, clock=None, **kwargs):
    clock = clock or FakeClock()
    return LLMClient(keys or all_keys(), budget=budget, transport=httpx.MockTransport(servers),
                     sleep=clock.sleep, clock=clock, **kwargs)


# ---------------------------------------------------------------------------
# Happy path and request shape
# ---------------------------------------------------------------------------

def test_first_provider_answers_and_usage_is_recorded():
    servers = FakeServers()
    client = make_client(servers)
    response = client.complete(MESSAGES, purpose="narrate")
    assert response.text == "All good."
    assert servers.called() == ["groq"]
    u = response.usage
    assert (u.provider, u.model, u.key_source, u.purpose) == ("groq", "openai/gpt-oss-120b", "server", "narrate")
    assert (u.prompt_tokens, u.completion_tokens, u.estimated, u.attempts) == (120, 30, False, 1)
    assert client.budget.used == 150


def test_request_is_openai_compatible_with_provider_settings():
    servers = FakeServers()
    make_client(servers).complete(MESSAGES, max_tokens=300, json_mode=True, tier="small")
    _, request = servers.requests[0]
    body = json.loads(request.content)
    assert request.url == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {GROQ_KEY}"
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"] == MESSAGES and body["max_tokens"] == 300
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "low"


def test_missing_usage_falls_back_to_estimates():
    servers = FakeServers(groq=[ok("short answer", usage=None)])
    response = make_client(servers).complete(MESSAGES)
    assert response.usage.estimated
    assert response.usage.prompt_tokens == estimate_messages(MESSAGES)
    assert response.usage.completion_tokens > 0


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

def test_rate_limit_without_short_retry_after_falls_back_and_cools_down():
    servers = FakeServers(groq=[httpx.Response(429, headers={"retry-after": "30"}), ok()])
    clock = FakeClock()
    client = make_client(servers, clock=clock)
    first = client.complete(MESSAGES)
    assert first.usage.provider == "cerebras"
    assert first.skipped == {"groq": "rate limited (429)"}
    assert clock.slept == []                          # 30 s is too long to wait

    second = client.complete(MESSAGES)                # groq is still cooling down: not even asked
    assert second.usage.provider == "cerebras"
    assert second.skipped["groq"] == "cooling down after a rate limit"
    assert servers.called() == ["groq", "cerebras", "cerebras"]

    clock.now += 31                                   # cooldown over: groq is tried again
    assert client.complete(MESSAGES).usage.provider == "groq"


def test_short_retry_after_is_waited_for_on_the_same_provider():
    servers = FakeServers(groq=[httpx.Response(429, headers={"retry-after": "1"}), ok()])
    clock = FakeClock()
    response = make_client(servers, clock=clock).complete(MESSAGES)
    assert response.usage.provider == "groq" and response.usage.attempts == 2
    assert clock.slept == [1.0]


def test_server_error_is_retried_once_then_falls_back():
    servers = FakeServers(groq=[httpx.Response(503), httpx.Response(502)])
    response = make_client(servers).complete(MESSAGES)
    assert servers.called() == ["groq", "groq", "cerebras"]
    assert response.skipped == {"groq": "server error (502)"}
    assert response.usage.attempts == 3


def test_transient_server_error_recovers_on_retry():
    servers = FakeServers(groq=[httpx.Response(500), ok()])
    response = make_client(servers).complete(MESSAGES)
    assert response.usage.provider == "groq" and servers.called() == ["groq", "groq"]


def test_timeouts_and_network_errors_fall_back():
    servers = FakeServers(groq=[httpx.ReadTimeout("slow")],
                          cerebras=[httpx.ConnectError("refused")])
    response = make_client(servers).complete(MESSAGES)
    assert response.usage.provider == "openrouter"
    assert response.skipped == {"groq": "timed out", "cerebras": "network error (ConnectError)"}


def test_rejected_key_falls_back_without_retrying():
    servers = FakeServers(groq=[httpx.Response(401, json={"error": {"message": "invalid key"}})])
    response = make_client(servers).complete(MESSAGES)
    assert servers.called() == ["groq", "cerebras"]
    assert response.skipped == {"groq": "key rejected (401)"}


def test_bad_request_falls_back_with_the_provider_message():
    servers = FakeServers(groq=[httpx.Response(400, json={"error": {"message": "model decommissioned"}})])
    response = make_client(servers).complete(MESSAGES)
    assert response.skipped["groq"] == "request refused (400): model decommissioned"
    assert servers.called() == ["groq", "cerebras"]


def test_empty_content_falls_back():
    servers = FakeServers(groq=[ok("   ")])
    assert make_client(servers).complete(MESSAGES).usage.provider == "cerebras"


def test_providers_without_keys_are_skipped():
    keys = KeyRing(server={"openrouter": ApiKey(OPENROUTER_KEY)})
    servers = FakeServers()
    response = make_client(servers, keys=keys).complete(MESSAGES)
    assert servers.called() == ["openrouter"]
    assert response.skipped == {"groq": "no API key", "cerebras": "no API key"}


def test_prompt_too_long_for_a_small_context_skips_that_provider():
    servers = FakeServers(groq=[httpx.Response(429)])
    big = [{"role": "user", "content": "word " * 8000}]
    response = make_client(servers, budget=TokenBudget(50_000)).complete(big, max_tokens=800)
    assert "cerebras" not in servers.called()          # 8k context on the free tier
    assert "context" in response.skipped["cerebras"]
    assert response.usage.provider == "openrouter"


def test_every_provider_failing_reports_each_reason():
    servers = FakeServers(groq=[httpx.Response(401)], cerebras=[httpx.Response(429)],
                          openrouter=[httpx.Response(500)])
    client = make_client(servers)
    with pytest.raises(AllProvidersFailed) as info:
        client.complete(MESSAGES)
    assert info.value.reasons == {"groq": "key rejected (401)", "cerebras": "rate limited (429)",
                                  "openrouter": "server error (500)"}
    assert client.budget.used == 0


def test_chain_order_is_configurable():
    servers = FakeServers()
    chain = [DEFAULT_CHAIN[2], DEFAULT_CHAIN[0]]
    response = make_client(servers, chain=chain).complete(MESSAGES)
    assert response.usage.provider == "openrouter"
    body = json.loads(servers.requests[0][1].content)
    assert body["model"] == "openai/gpt-oss-120b:free" and body["reasoning"] == {"effort": "low"}


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

def test_budget_is_checked_before_any_request():
    servers = FakeServers()
    client = make_client(servers, budget=TokenBudget(500))
    with pytest.raises(BudgetExceeded, match="only 500 of 500 remain"):
        client.complete(MESSAGES, max_tokens=800, purpose="narrate")
    assert servers.requests == []


def test_budget_accumulates_across_calls_and_then_refuses():
    servers = FakeServers(groq=[ok(usage=(3000, 1000))])
    client = make_client(servers, budget=TokenBudget(10_000))
    client.complete(MESSAGES, max_tokens=1000, purpose="plan")
    client.complete(MESSAGES, max_tokens=1000, purpose="narrate")
    assert client.budget.used == 8000 and client.budget.remaining == 2000
    with pytest.raises(BudgetExceeded):
        client.complete(MESSAGES, max_tokens=2000, purpose="critic")
    summary = client.budget.summary()
    assert summary["calls"] == 2 and summary["by_provider"] == {"groq": 8000}
    assert [c["purpose"] for c in summary["detail"]] == ["plan", "narrate"]


def test_default_run_budget_is_10k():
    assert make_client(FakeServers()).budget.limit == 10_000


# ---------------------------------------------------------------------------
# Bring-your-own-key safety
# ---------------------------------------------------------------------------

def test_user_key_overrides_server_key_and_is_marked_byok():
    servers = FakeServers()
    keys = all_keys().with_user_keys({"groq": "gsk_user_own_key_9"})
    response = make_client(servers, keys=keys).complete(MESSAGES)
    assert servers.requests[0][1].headers["authorization"] == "Bearer gsk_user_own_key_9"
    assert response.usage.key_source == "user"


def test_user_keys_do_not_change_the_server_ring():
    server = all_keys()
    server.with_user_keys({"groq": "gsk_user_own_key_9"})
    assert server.get("groq").reveal() == GROQ_KEY and server.source("groq") == "server"


def test_keys_never_appear_in_repr_str_or_format():
    key = ApiKey(GROQ_KEY)
    ring = all_keys().with_user_keys({"cerebras": "csk_user_key_7"})
    for text in (repr(key), str(key), f"{key}", f"{key!r}", repr(ring), str([key]), str({"k": key})):
        assert GROQ_KEY not in text and "csk_user_key_7" not in text
    assert repr(ring) == "KeyRing(cerebras=user, groq=server, openrouter=server)"


def test_keys_cannot_be_pickled():
    with pytest.raises(TypeError):
        pickle.dumps(ApiKey(GROQ_KEY))


def test_empty_keys_are_rejected_or_ignored():
    with pytest.raises(ValueError):
        ApiKey("  ")
    assert all_keys().with_user_keys({"groq": ""}).source("groq") == "server"


def test_keys_are_scrubbed_from_errors_and_logs(caplog):
    leaky = httpx.Response(400, json={"error": {"message": f"Invalid API key: {GROQ_KEY}"}})
    servers = FakeServers(groq=[leaky], cerebras=[httpx.Response(401)], openrouter=[httpx.Response(500)])
    with caplog.at_level(logging.DEBUG), pytest.raises(AllProvidersFailed) as info:
        make_client(servers).complete(MESSAGES)
    assert GROQ_KEY not in str(info.value)
    assert "Invalid API key: ****" in info.value.reasons["groq"]
    assert GROQ_KEY not in caplog.text and "****" in caplog.text


def test_redact_handles_overlapping_secrets():
    assert redact("a abc ab", ["ab", "abc"]) == "a **** ****"


def test_client_from_env_reads_server_keys_and_applies_user_keys(monkeypatch):
    for var in KEY_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    client = client_from_env({"openrouter": OPENROUTER_KEY}, budget=5000)
    assert client.keys.source("groq") == "server"
    assert client.keys.source("openrouter") == "user"
    assert client.keys.get("cerebras") is None
    assert client.budget.limit == 5000


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Here you go: {"a": 1} hope that helps',
])
def test_parse_json_tolerates_fences_and_prose(text):
    assert parse_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["no json here", "[1, 2]", '{"a": }'])
def test_parse_json_rejects_non_objects(text):
    with pytest.raises(LLMOutputError):
        parse_json(text)


def test_complete_json_asks_once_more_on_invalid_output():
    servers = FakeServers(groq=[ok("sorry, here is text"), ok('{"verdict": "pass"}')])
    client = make_client(servers)
    data, response = client.complete_json(MESSAGES, purpose="critic")
    assert data == {"verdict": "pass"}
    retry_body = json.loads(servers.requests[1][1].content)
    assert retry_body["messages"][-1]["content"].startswith("That was not valid JSON")
    assert client.budget.summary()["calls"] == 2       # the retry is paid for, and counted


def test_complete_json_gives_up_after_one_retry():
    servers = FakeServers(groq=[ok("not json")])
    with pytest.raises(LLMOutputError):
        make_client(servers).complete_json(MESSAGES)
    assert len(servers.requests) == 2
