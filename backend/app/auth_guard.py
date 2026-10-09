"""인증 — 비밀번호 해시 + 로그인 토큰(JWT) 발급/검증 의존성.

토큰은 서버 비밀키로 서명한 JWT(HS256)라 위조할 수 없고, AUTH_TOKEN_TTL_HOURS 가
지나면 만료된다. 예전 "tok_{user_id}" 토큰은 사용자 번호만 바꾸면 다른 사람으로
행세할 수 있었다.
NOTE: 비밀번호 해시(salt 없는 SHA-256) 강화는 남은 과제.
"""
import hashlib
from datetime import datetime, timedelta, timezone

import jwt

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models
from app.settings import settings, auth_secret_key


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


_JWT_ALG = "HS256"


def create_token(user: "db_models.User") -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "iat": now,
        "exp": now + timedelta(hours=settings.AUTH_TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, auth_secret_key(), algorithm=_JWT_ALG)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db),
) -> "db_models.User":
    """Authorization: Bearer <token> 헤더에서 현재 사용자를 해석한다."""
    if not authorization:
        raise _unauthorized("인증이 필요합니다.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        # 알고리즘을 고정해 'alg' 바꿔치기 공격을 막고, 만료·대상 필드를 필수로 요구한다
        payload = jwt.decode(token, auth_secret_key(), algorithms=[_JWT_ALG],
                             options={"require": ["exp", "sub"]})
        user_id = int(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise _unauthorized("로그인이 만료되었습니다. 다시 로그인해 주세요.")
    except (jwt.InvalidTokenError, ValueError):
        raise _unauthorized("유효하지 않은 토큰입니다.")
    user = db.get(db_models.User, user_id)
    if user is None:
        raise _unauthorized("사용자를 찾을 수 없습니다.")
    return user
