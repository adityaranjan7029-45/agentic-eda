"""LLM providers: all three speak the OpenAI-compatible chat-completions API,
so one HTTP client covers them; only base URL, model IDs and limits differ.

Free tiers are small (Groq: about 8,000 tokens per minute for gpt-oss-120b),
so a 10k-token run can hit rate limits. The fallback chain handles that.
Model IDs change over time; they live here, in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    models: dict[str, str]          # tier -> model id ("large" for reasoning, "small" for cheap tasks)
    key_env: str                    # environment variable holding the server's key
    context_window: int             # tokens (prompt + completion) the model accepts
    extra_body: dict = field(default_factory=dict)
    extra_headers: dict = field(default_factory=dict)

    def model(self, tier: str) -> str:
        return self.models.get(tier) or self.models["large"]


GROQ = ProviderSpec(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    models={"large": "openai/gpt-oss-120b", "small": "openai/gpt-oss-20b"},
    key_env="GROQ_API_KEY",
    context_window=131_072,
    extra_body={"reasoning_effort": "low"},
)

CEREBRAS = ProviderSpec(
    name="cerebras",
    base_url="https://api.cerebras.ai/v1",
    models={"large": "gpt-oss-120b"},
    key_env="CEREBRAS_API_KEY",
    context_window=8_192,           # free tier limit
    extra_body={"reasoning_effort": "low"},
)

OPENROUTER = ProviderSpec(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    models={"large": "openai/gpt-oss-120b:free", "small": "openai/gpt-oss-20b:free"},
    key_env="OPENROUTER_API_KEY",
    context_window=131_072,
    extra_body={"reasoning": {"effort": "low"}},
    extra_headers={"X-Title": "Agentic EDA"},
)

PROVIDERS = {p.name: p for p in (GROQ, CEREBRAS, OPENROUTER)}
DEFAULT_CHAIN = (GROQ, CEREBRAS, OPENROUTER)
KEY_ENV_VARS = {p.name: p.key_env for p in PROVIDERS.values()}
