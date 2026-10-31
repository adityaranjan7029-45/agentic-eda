"""LLM provider layer: fallback chain, bring-your-own-key safety, token budget."""

from eda_engine.llm.budget import DEFAULT_RUN_BUDGET, BudgetExceeded, TokenBudget, Usage, estimate_messages
from eda_engine.llm.client import (
    AllProvidersFailed,
    LLMClient,
    LLMError,
    LLMOutputError,
    LLMResponse,
    parse_json,
)
from eda_engine.llm.keys import ApiKey, KeyRing, redact
from eda_engine.llm.providers import DEFAULT_CHAIN, KEY_ENV_VARS, PROVIDERS, ProviderSpec


def client_from_env(user_keys: dict[str, str] | None = None, budget: int = DEFAULT_RUN_BUDGET, **kwargs) -> LLMClient:
    """Server keys from the environment, overridden by the user's own keys for this run."""
    keys = KeyRing.from_env(KEY_ENV_VARS).with_user_keys(user_keys or {})
    return LLMClient(keys, budget=TokenBudget(budget), **kwargs)


__all__ = [
    "DEFAULT_CHAIN", "DEFAULT_RUN_BUDGET", "KEY_ENV_VARS", "PROVIDERS",
    "AllProvidersFailed", "ApiKey", "BudgetExceeded", "KeyRing", "LLMClient", "LLMError",
    "LLMOutputError", "LLMResponse", "ProviderSpec", "TokenBudget", "Usage",
    "client_from_env", "estimate_messages", "parse_json", "redact",
]
