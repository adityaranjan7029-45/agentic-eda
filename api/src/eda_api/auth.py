"""Who is calling.

Production: the Next.js app (Auth.js, GitHub login) signs a short-lived token for
every API call: HS256 with API_JWT_SECRET, issuer "agentic-eda-web", audience
"agentic-eda-api", `sub` = "github:<account id>", plus email and name. The browser
never sees this token; it only holds Auth.js's httpOnly session cookie, and the
Next.js server forwards its requests here (backend-for-frontend).

Verification is strict: the algorithm is pinned (no "alg: none", no RS/HS confusion),
issuer, audience, expiry and issued-at are required, and tokens that claim to live
longer than JWT_MAX_LIFETIME_SECONDS are refused even if correctly signed.

Dev/test only: `X-User-Email` still works for quick testing with /docs. In prod it is ignored.
"""

from __future__ import annotations

import re

import jwt
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from eda_api.db import User

EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
SUBJECT = re.compile(r"^(?P<provider>[a-z][a-z0-9-]{1,31}):(?P<id>[A-Za-z0-9._-]{1,255})$")
ALGORITHMS = ["HS256"]
LEEWAY_SECONDS = 30


def get_session(request: Request):
    with request.app.state.sessionmaker() as session:
        yield session


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def verify_token(token: str, settings) -> dict:
    if not settings.api_jwt_secret:
        raise _unauthorized("token login is not configured on this server")
    try:
        claims = jwt.decode(
            token, settings.api_jwt_secret.get_secret_value(), algorithms=ALGORITHMS,
            audience=settings.jwt_audience, issuer=settings.jwt_issuer, leeway=LEEWAY_SECONDS,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("token expired") from None
    except jwt.InvalidTokenError:
        raise _unauthorized("invalid token") from None
    if claims["exp"] - claims["iat"] > settings.jwt_max_lifetime_seconds:
        raise _unauthorized("token lifetime too long")
    if not isinstance(claims["sub"], str) or not SUBJECT.match(claims["sub"]):
        raise _unauthorized("invalid token subject")
    return claims


def _upsert(session: Session, provider: str, subject: str, email: str | None, name: str | None) -> User:
    user = session.scalar(select(User).where(User.auth_provider == provider, User.auth_subject == subject))
    if user is None:
        user = User(auth_provider=provider, auth_subject=subject, email=email, name=name)
        session.add(user)
        session.commit()
    elif (email and user.email != email) or (name and user.name != name):
        user.email, user.name = email or user.email, name or user.name   # keep profile details current
        session.commit()
    return user


def current_user(request: Request, session: Session = Depends(get_session),
                 authorization: str | None = Header(default=None),
                 x_user_email: str | None = Header(default=None)) -> User:
    settings = request.app.state.settings
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise _unauthorized("use: Authorization: Bearer <token>")
        claims = verify_token(token.strip(), settings)
        match = SUBJECT.match(claims["sub"])
        email = claims.get("email") if isinstance(claims.get("email"), str) else None
        name = claims.get("name") if isinstance(claims.get("name"), str) else None
        return _upsert(session, match["provider"], match["id"], (email or "").lower()[:320] or None,
                       (name or "")[:255] or None)

    if settings.dev_auth:
        email = (x_user_email or "").strip().lower()
        if EMAIL.match(email):
            return _upsert(session, "dev", email, email, None)
        raise _unauthorized("sign in, or send X-User-Email (dev mode only)")
    raise _unauthorized("sign in required")
