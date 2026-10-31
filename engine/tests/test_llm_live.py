"""Optional smoke test against the real providers.

Skipped unless a key is set, so CI never calls a paid or rate-limited API.
Run locally:  $env:GROQ_API_KEY="gsk_..."; python -m pytest engine/tests/test_llm_live.py -v
"""

import os

import pytest

from eda_engine.llm import KEY_ENV_VARS, PROVIDERS, KeyRing, LLMClient, TokenBudget

AVAILABLE = [name for name, var in KEY_ENV_VARS.items() if os.environ.get(var, "").strip()]


@pytest.mark.skipif(not AVAILABLE, reason="no provider API key in the environment")
@pytest.mark.parametrize("provider", AVAILABLE)
def test_real_provider_answers_json_within_budget(provider):
    client = LLMClient(KeyRing.from_env(KEY_ENV_VARS), chain=[PROVIDERS[provider]], budget=TokenBudget(2000))
    data, response = client.complete_json(
        [{"role": "user", "content": 'Reply with exactly this JSON object: {"status": "ok"}'}],
        max_tokens=300, purpose="smoke")
    assert data.get("status") == "ok"
    assert response.usage.provider == provider and not response.usage.estimated
    assert client.budget.used <= 2000
