"""Token accounting: every run has a hard token budget (10,000 by default).

Before each call the client estimates prompt + maximum completion tokens and
refuses the call if it could overrun the budget, so a run can never cost
more than its budget. After the call, the provider's real usage numbers are
recorded (estimates are used only if a provider omits them).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from eda_engine.tokens import estimate_tokens

DEFAULT_RUN_BUDGET = 10_000
MESSAGE_OVERHEAD = 4          # role markers and separators per chat message


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Usage:
    purpose: str              # what the call was for, e.g. "narrate" or "critic"
    provider: str
    model: str
    key_source: str           # "user" (BYOK) or "server"
    prompt_tokens: int
    completion_tokens: int
    estimated: bool           # True if the provider didn't report usage
    latency_s: float
    attempts: int             # HTTP requests made, including retries and fallbacks

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_messages(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) + MESSAGE_OVERHEAD for m in messages)


@dataclass
class TokenBudget:
    limit: int = DEFAULT_RUN_BUDGET
    calls: list[Usage] = field(default_factory=list)

    @property
    def used(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def check(self, prompt_estimate: int, max_completion: int, purpose: str = "") -> None:
        needed = prompt_estimate + max_completion
        if needed > self.remaining:
            raise BudgetExceeded(
                f"{purpose or 'call'} needs up to {needed:,} tokens "
                f"(~{prompt_estimate:,} prompt + {max_completion:,} completion) but only "
                f"{self.remaining:,} of {self.limit:,} remain"
            )

    def record(self, usage: Usage) -> None:
        self.calls.append(usage)

    def summary(self) -> dict:
        by_provider: dict[str, int] = {}
        for c in self.calls:
            by_provider[c.provider] = by_provider.get(c.provider, 0) + c.total_tokens
        return {
            "limit": self.limit, "used": self.used, "remaining": self.remaining,
            "calls": len(self.calls), "by_provider": by_provider,
            "detail": [{**asdict(c), "total_tokens": c.total_tokens} for c in self.calls],
        }
