"""간소 인증 — 비밀번호 해시 + 토큰 발급/검증 의존성.

NOTE: 보안 강화는 추후 과제. 현재는 MVP 동작을 위한 최소 구현.
토큰은 단순히 "tok_{user_id}" 형태이며 서명/만료가 없다.
"""
import hashlib

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


def create_token(user: "db_models.User") -> str:
    return f"tok_{user.id}"


def _parse_token(token: str) -> int:
    if not token or not token.startswith("tok_"):
        raise ValueError("invalid token")
    return int(token[len("tok_"):])


def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db),
) -> "db_models.User":
    """Authorization: Bearer <token> 헤더에서 현재 사용자를 해석한다."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
        )
    token = authorization.removeprefix("Bearer ").strip()
    try:
        user_id = _parse_token(token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
        )
    user = db.get(db_models.User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다.",
        )
    return user
