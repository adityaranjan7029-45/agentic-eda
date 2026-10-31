"""One LLM client with a provider fallback chain.

For each call it walks the chain (Groq -> Cerebras -> OpenRouter by default):

  - no key for a provider, or provider cooling down, or prompt too long for
    its context window -> skip it
  - 429 rate limit -> wait and retry once if the provider asks for a short
    wait, otherwise put it on cooldown and fall back
  - 5xx / timeout / network error -> retry once after a short backoff, then fall back
  - 401/403 (bad key) and other 4xx -> fall back immediately (no retry)
  - 200 with empty content -> fall back

If every provider fails, `AllProvidersFailed` lists why for each one, with
keys scrubbed out. The token budget is checked before any request is made.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx

from eda_engine.llm.budget import TokenBudget, Usage, estimate_messages
from eda_engine.llm.keys import KeyRing, redact
from eda_engine.llm.providers import DEFAULT_CHAIN, ProviderSpec
from eda_engine.tokens import estimate_tokens

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class AllProvidersFailed(LLMError):
    def __init__(self, reasons: dict[str, str]):
        self.reasons = reasons
        super().__init__("all LLM providers failed: " + "; ".join(f"{p}: {r}" for p, r in reasons.items()))


class LLMOutputError(LLMError):
    """The model answered, but not in the format asked for."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage
    skipped: dict[str, str]       # providers tried or skipped before the one that answered, with reasons


class _ProviderFailed(Exception):
    def __init__(self, reason: str, cooldown: float | None = None, attempts: int = 1):
        super().__init__(reason)
        self.reason, self.cooldown, self.attempts = reason, cooldown, attempts


class LLMClient:
    def __init__(
        self,
        keys: KeyRing,
        chain: Sequence[ProviderSpec] = DEFAULT_CHAIN,
        budget: TokenBudget | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        max_wait: float = 3.0,            # longest Retry-After worth waiting for before falling back
        default_cooldown: float = 60.0,   # how long a rate-limited provider is skipped
        backoff: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.keys, self.chain = keys, list(chain)
        self.budget = budget if budget is not None else TokenBudget()
        self.max_wait, self.default_cooldown, self.backoff = max_wait, default_cooldown, backoff
        self._sleep, self._clock = sleep, clock
        self._cooldown_until: dict[str, float] = {}
        self._http = httpx.Client(transport=transport, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------

    def complete(self, messages: list[dict], *, tier: str = "large", max_tokens: int = 800,
                 temperature: float = 0.2, json_mode: bool = False, purpose: str = "") -> LLMResponse:
        prompt_estimate = estimate_messages(messages)
        self.budget.check(prompt_estimate, max_tokens, purpose)

        reasons: dict[str, str] = {}
        attempts = 0
        for spec in self.chain:
            key = self.keys.get(spec.name)
            if key is None:
                reasons[spec.name] = "no API key"
                continue
            if self._cooldown_until.get(spec.name, 0.0) > self._clock():
                reasons[spec.name] = "cooling down after a rate limit"
                continue
            if prompt_estimate + max_tokens > spec.context_window:
                reasons[spec.name] = f"prompt too long for its {spec.context_window:,}-token context"
                continue

            body = {"model": spec.model(tier), "messages": messages, "max_tokens": max_tokens,
                    "temperature": temperature, **spec.extra_body}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            headers = {"Authorization": f"Bearer {key.reveal()}", **spec.extra_headers}

            started = self._clock()
            try:
                data, n = self._post_with_retry(spec, body, headers)
            except _ProviderFailed as failure:
                attempts += failure.attempts
                reasons[spec.name] = redact(failure.reason, self.keys.secrets())
                if failure.cooldown:
                    self._cooldown_until[spec.name] = self._clock() + failure.cooldown
                log.warning("LLM provider %s failed (%s); falling back", spec.name, reasons[spec.name])
                continue
            attempts += n

            text = _content(data)
            if not text.strip():
                reasons[spec.name] = "empty response"
                continue
            reported = data.get("usage") or {}
            usage = Usage(
                purpose=purpose, provider=spec.name, model=str(data.get("model") or body["model"]),
                key_source=self.keys.source(spec.name) or "unknown",
                prompt_tokens=int(reported.get("prompt_tokens") or prompt_estimate),
                completion_tokens=int(reported.get("completion_tokens") or estimate_tokens(text)),
                estimated=not reported, latency_s=round(self._clock() - started, 3), attempts=attempts,
            )
            self.budget.record(usage)
            return LLMResponse(text=text, usage=usage, skipped=reasons)

        raise AllProvidersFailed(reasons)

    def complete_json(self, messages: list[dict], **kwargs) -> tuple[dict, LLMResponse]:
        """Ask for JSON; if the reply doesn't parse, ask once more, then give up."""
        response = self.complete(messages, json_mode=True, **kwargs)
        try:
            return parse_json(response.text), response
        except LLMOutputError:
            retry = [*messages, {"role": "assistant", "content": response.text},
                     {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."}]
            response = self.complete(retry, json_mode=True, **kwargs)
            return parse_json(response.text), response

    # ------------------------------------------------------------------

    def _post_with_retry(self, spec: ProviderSpec, body: dict, headers: dict) -> tuple[dict, int]:
        """POST once, retrying once on transient errors. Returns (json, attempts)."""
        url = f"{spec.base_url}/chat/completions"
        for attempt in (1, 2):
            try:
                response = self._http.post(url, json=body, headers=headers)
            except httpx.TimeoutException:
                reason = "timed out"
            except httpx.TransportError as exc:
                reason = f"network error ({type(exc).__name__})"
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return response.json(), attempt
                    except ValueError:
                        raise _ProviderFailed("invalid JSON from provider", attempts=attempt) from None
                if status == 429:
                    wait = _retry_after(response)
                    if attempt == 1 and wait is not None and wait <= self.max_wait:
                        self._sleep(wait)
                        continue
                    raise _ProviderFailed("rate limited (429)", cooldown=wait or self.default_cooldown,
                                          attempts=attempt)
                if status in (401, 403):
                    raise _ProviderFailed(f"key rejected ({status})", attempts=attempt)
                if status not in RETRYABLE_STATUS:
                    raise _ProviderFailed(f"request refused ({status}): {_error_message(response)}",
                                          attempts=attempt)
                reason = f"server error ({status})"
            if attempt == 1:
                self._sleep(self.backoff)
        raise _ProviderFailed(reason, attempts=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content(data: dict) -> str:
    try:
        return data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        message = error.get("message") if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        message = response.text
    return (message or "")[:200]


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and stray prose."""
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError("no JSON object in model output") from None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"invalid JSON in model output: {exc.msg}") from None
    if not isinstance(value, dict):
        raise LLMOutputError("model output is JSON but not an object")
    return value
