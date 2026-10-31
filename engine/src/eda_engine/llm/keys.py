"""API keys that can't leak by accident.

Bring-your-own-key (BYOK) rule: a user's key lives only in memory for the
duration of one run. It is never written to disk, a database, a log or an
error message. These classes make the safe path the default one:

  - `ApiKey` hides its value from repr/str/f-strings; only `reveal()` returns
    it, and only the HTTP client calls that, to build the Authorization header.
  - `KeyRing` resolves which key to use per provider: the user's own key wins,
    otherwise the server's key from the environment.
  - `redact()` scrubs any key value out of text before it is logged or raised.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


class ApiKey:
    __slots__ = ("_value",)

    def __init__(self, value: str):
        value = (value or "").strip()
        if not value:
            raise ValueError("empty API key")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "ApiKey(****)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ApiKey) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __reduce__(self):
        # Pickling would copy the secret into caches, queues or files.
        raise TypeError("ApiKey cannot be pickled")


class KeyRing:
    """Keys per provider, with their source: 'user' (BYOK) or 'server' (env)."""

    def __init__(self, server: Mapping[str, ApiKey] | None = None, user: Mapping[str, ApiKey] | None = None):
        self._server = dict(server or {})
        self._user = dict(user or {})

    @classmethod
    def from_env(cls, env_vars: Mapping[str, str], env: Mapping[str, str] | None = None) -> KeyRing:
        """env_vars: provider -> environment variable name, e.g. {"groq": "GROQ_API_KEY"}."""
        env = os.environ if env is None else env
        return cls(server={p: ApiKey(env[v]) for p, v in env_vars.items() if env.get(v, "").strip()})

    def with_user_keys(self, keys: Mapping[str, str | ApiKey]) -> KeyRing:
        """A new ring where the user's keys override the server's (for one run)."""
        user = dict(self._user)
        for provider, key in keys.items():
            if isinstance(key, ApiKey):
                user[provider] = key
            elif key and key.strip():
                user[provider] = ApiKey(key)
        return KeyRing(self._server, user)

    def get(self, provider: str) -> ApiKey | None:
        return self._user.get(provider) or self._server.get(provider)

    def source(self, provider: str) -> str | None:
        if provider in self._user:
            return "user"
        return "server" if provider in self._server else None

    def secrets(self) -> list[str]:
        return [k.reveal() for k in (*self._user.values(), *self._server.values())]

    def __repr__(self) -> str:
        providers = sorted(set(self._user) | set(self._server))
        return f"KeyRing({', '.join(f'{p}={self.source(p)}' for p in providers)})"


def redact(text: str, secrets: list[str]) -> str:
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            text = text.replace(secret, "****")
    return text
