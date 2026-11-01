"""Who is calling. Step 11: a dev-only header. Step 13 replaces this with Auth.js sessions.

In dev/test the caller sends `X-User-Email`; the user row is created on first
use. In prod this dependency refuses every request until real auth exists,
so the header can never be used to impersonate someone on the live server.
"""

from __future__ import annotations

import re

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from eda_api.db import User

EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")


def get_session(request: Request):
    with request.app.state.sessionmaker() as session:
        yield session


def current_user(request: Request, session: Session = Depends(get_session),
                 x_user_email: str | None = Header(default=None)) -> User:
    if not request.app.state.settings.dev_auth:
        raise HTTPException(401, "authentication is not configured yet")
    email = (x_user_email or "").strip().lower()
    if not EMAIL.match(email):
        raise HTTPException(401, "send X-User-Email (dev mode only)")
    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email)
        session.add(user)
        session.commit()
    return user
