"""Login: tokens signed by the Next.js app (Auth.js) and verified by the API."""

import base64
import json
import time

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from eda_api.app import create_app
from eda_api.config import Settings

SECRET = "jwt-test-secret-that-is-long-enough-for-every-hmac-variant-0123456789"
CLAIMS = {"iss": "agentic-eda-web", "aud": "agentic-eda-api", "sub": "github:583231",
          "email": "Aditya@Example.com", "name": "Aditya"}


def token(secret=SECRET, algorithm="HS256", lifetime=300, issued_ago=0, **overrides):
    now = int(time.time()) - issued_ago
    claims = {**CLAIMS, "iat": now, "exp": now + lifetime, **overrides}
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, secret, algorithm=algorithm)


def bearer(t):
    return {"Authorization": f"Bearer {t}"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL", "REDIS_URL",
                "APP_ENV", "API_JWT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None, app_env="test", database_url="sqlite:///:memory:",
                    upload_dir=tmp_path / "u", inline_workers=0, api_jwt_secret=SECRET)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def prod_client(settings):
    prod = settings.model_copy(update={"app_env": "prod"})   # sqlite is fine for the test; skip prod URL rule
    with TestClient(create_app(prod)) as c:
        yield c


def me(client, headers):
    """GET /runs succeeds only for an authenticated user; returns the status code."""
    return client.get("/runs", headers=headers).status_code


# ---------------------------------------------------------------------------
# Valid tokens
# ---------------------------------------------------------------------------

def test_valid_token_is_accepted_and_creates_the_user(client):
    assert me(client, bearer(token())) == 200
    with client.app.state.sessionmaker() as s:
        from eda_api.db import User
        user = s.query(User).one()
    assert (user.auth_provider, user.auth_subject, user.email, user.name) == ("github", "583231",
                                                                              "aditya@example.com", "Aditya")


def test_same_account_is_one_user_even_if_email_changes(client):
    me(client, bearer(token()))
    me(client, bearer(token(email="new@example.com")))
    with client.app.state.sessionmaker() as s:
        from eda_api.db import User
        users = s.query(User).all()
    assert len(users) == 1 and users[0].email == "new@example.com"


def test_two_accounts_with_the_same_email_are_different_users(client):
    me(client, bearer(token()))
    me(client, bearer(token(sub="github:999")))
    with client.app.state.sessionmaker() as s:
        from eda_api.db import User
        assert s.query(User).count() == 2


def test_google_accounts_work_and_are_separate_from_github(client):
    """Google sign-in (sub = Google's 21-digit account id) is just another provider:
    the same email on GitHub and Google is two users, each seeing only their own runs."""
    from eda_api.db import User

    google = bearer(token(sub="google:109876543210987654321"))
    assert me(client, google) == 200
    assert me(client, bearer(token())) == 200            # github:583231, same email
    with client.app.state.sessionmaker() as s:
        identities = sorted((u.auth_provider, u.auth_subject) for u in s.query(User).all())
    assert identities == [("github", "583231"), ("google", "109876543210987654321")]


def test_missing_email_is_fine(client):
    assert me(client, bearer(token(email=None))) == 200


def test_runs_are_private_per_github_account(client):
    from eda_engine.synth.credit import generate_credit_applications
    import io
    csv = generate_credit_applications(n_rows=800, seed=1).data.to_csv(index=False).encode()
    run = client.post("/runs", files={"file": ("c.csv", io.BytesIO(csv), "text/csv")},
                      headers=bearer(token())).json()
    assert client.get(f"/runs/{run['id']}", headers=bearer(token())).status_code == 200
    assert client.get(f"/runs/{run['id']}", headers=bearer(token(sub="github:999"))).status_code == 404


def test_clock_skew_within_30_seconds_is_tolerated(client):
    assert me(client, bearer(token(lifetime=300, issued_ago=320))) == 200      # expired 20 s ago


# ---------------------------------------------------------------------------
# Rejected tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad, reason", [
    (lambda: token(secret="another-secret-that-is-also-long-enough-0123456"), "invalid token"),
    (lambda: token(lifetime=300, issued_ago=400), "token expired"),
    (lambda: token(aud="someone-else"), "invalid token"),
    (lambda: token(iss="evil-site"), "invalid token"),
    (lambda: token(exp=None), "invalid token"),
    (lambda: token(iat=None), "invalid token"),
    (lambda: token(sub=None), "invalid token"),
    (lambda: token(lifetime=86_400), "token lifetime too long"),
    (lambda: token(sub="583231"), "invalid token subject"),
    (lambda: token(sub="github:../../etc"), "invalid token subject"),
    (lambda: token(algorithm="HS512"), "invalid token"),
    (lambda: "not.a.jwt", "invalid token"),
])
def test_bad_tokens_are_refused(client, bad, reason):
    response = client.get("/runs", headers=bearer(bad()))
    assert response.status_code == 401 and response.json()["detail"] == reason
    assert response.headers["www-authenticate"] == "Bearer"


def test_alg_none_is_refused(client):
    """The classic JWT attack: an unsigned token that says 'alg: none'."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    now = int(time.time())
    body = base64.urlsafe_b64encode(json.dumps({**CLAIMS, "iat": now, "exp": now + 300}).encode()).rstrip(b"=")
    unsigned = f"{header.decode()}.{body.decode()}."
    assert me(client, bearer(unsigned)) == 401


@pytest.mark.parametrize("header", ["Basic abc", "Bearer", "Token abc"])
def test_malformed_authorization_header(client, header):
    assert me(client, {"Authorization": header}) == 401


def test_tokens_refused_when_server_has_no_secret(settings):
    with TestClient(create_app(settings.model_copy(update={"api_jwt_secret": None}))) as c:
        response = c.get("/runs", headers=bearer(token()))
    assert response.status_code == 401 and "not configured" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Production rules
# ---------------------------------------------------------------------------

def test_prod_ignores_the_dev_header_but_accepts_tokens(prod_client):
    assert me(prod_client, {"X-User-Email": "alice@example.com"}) == 401
    assert me(prod_client, {}) == 401
    assert me(prod_client, bearer(token())) == 200


def test_dev_header_still_works_in_dev_and_is_a_separate_identity(client):
    assert me(client, {"X-User-Email": "aditya@example.com"}) == 200
    me(client, bearer(token()))          # same email via GitHub: a different account
    with client.app.state.sessionmaker() as s:
        from eda_api.db import User
        assert {u.auth_provider for u in s.query(User).all()} == {"dev", "github"}


def test_settings_require_a_long_jwt_secret_in_prod(monkeypatch):
    for var in ("API_JWT_SECRET", "APP_ENV", "DATABASE_URL", "REDIS_URL"):
        monkeypatch.delenv(var, raising=False)
    pg = "postgresql://u:p@h/db"
    with pytest.raises(ValueError, match="API_JWT_SECRET"):
        Settings(_env_file=None, app_env="prod", database_url=pg)
    with pytest.raises(ValueError, match="at least 32"):
        Settings(_env_file=None, database_url=pg, api_jwt_secret=SecretStr("short"))
    assert Settings(_env_file=None, app_env="prod", database_url=pg, api_jwt_secret=SECRET).dev_auth is False
