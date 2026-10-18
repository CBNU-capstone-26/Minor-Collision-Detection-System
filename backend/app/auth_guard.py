"""인증 — 비밀번호 해시 + 로그인 토큰(JWT) 발급/검증 의존성.

토큰은 서버 비밀키로 서명한 JWT(HS256)라 위조할 수 없고, AUTH_TOKEN_TTL_HOURS 가
지나면 만료된다. 예전 "tok_{user_id}" 토큰은 사용자 번호만 바꾸면 다른 사람으로
행세할 수 있었다.

영상·썸네일·클립은 <video>·<img>·내려받기 링크로 열려서 Authorization 헤더를 실을 수 없다.
그래서 로그인할 때 '영상 전용 토큰'을 HttpOnly 쿠키로 함께 준다(get_media_user).
이 토큰은 영상 주소에서만 통하고 다른 API 에선 거절된다 — 새도 피해가 영상 보기로 한정된다.
"""
import functools
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from fastapi import Depends, Header, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models
from app.settings import settings, auth_secret_key


# ── 비밀번호 ───────────────────────────────────────────────────────────
# PBKDF2-SHA256, 사용자마다 다른 소금(16바이트), 60만 회(OWASP 2023 권장).
# 저장 형식: pbkdf2_sha256$회수$소금hex$해시hex
# 예전 형식(소금 없는 SHA-256 hex)도 확인은 해 주고, 로그인에 성공하면 새 형식으로 바꿔 둔다.
_PBKDF2_PREFIX = "pbkdf2_sha256"
_PBKDF2_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{_PBKDF2_PREFIX}${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith(_PBKDF2_PREFIX + "$"):
        try:
            _, iterations, salt_hex, digest_hex = password_hash.split("$")
            digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                         bytes.fromhex(salt_hex), int(iterations))
        except ValueError:
            return False
        return hmac.compare_digest(digest.hex(), digest_hex)
    legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()   # 예전 형식
    return hmac.compare_digest(legacy, password_hash)


def needs_rehash(password_hash: str) -> bool:
    """예전 형식이거나 반복 회수가 지금 기준과 다르면 True."""
    return not password_hash.startswith(f"{_PBKDF2_PREFIX}${_PBKDF2_ITERATIONS}$")


@functools.lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_password(secrets.token_hex(8))


def burn_password_check(password: str) -> None:
    """없는 아이디로 로그인할 때도 비밀번호 확인만큼 시간을 쓴다 — 응답 시간으로 아이디가
    있는지 알아내지 못하게."""
    verify_password(password, _dummy_hash())


# ── 로그인 토큰 ─────────────────────────────────────────────────────────
_JWT_ALG = "HS256"
_MEDIA_SCOPE = "media"
MEDIA_COOKIE = "siot_media"


def create_token(user: "db_models.User", scope: str | None = None,
                 expires_at: datetime | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "iat": now,
        "exp": expires_at or now + timedelta(hours=settings.AUTH_TOKEN_TTL_HOURS),
    }
    if scope:
        payload["scope"] = scope
    return jwt.encode(payload, auth_secret_key(), algorithm=_JWT_ALG)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _decode(token: str) -> dict:
    try:
        # 알고리즘을 고정해 'alg' 바꿔치기 공격을 막고, 만료·대상 필드를 필수로 요구한다
        payload = jwt.decode(token, auth_secret_key(), algorithms=[_JWT_ALG],
                             options={"require": ["exp", "sub"]})
        int(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise _unauthorized("로그인이 만료되었습니다. 다시 로그인해 주세요.")
    except (jwt.InvalidTokenError, ValueError):
        raise _unauthorized("유효하지 않은 토큰입니다.")
    return payload


def bearer_payload(authorization: str | None) -> dict:
    """Authorization: Bearer <token> 헤더의 일반 로그인 토큰 내용."""
    if not authorization:
        raise _unauthorized("인증이 필요합니다.")
    payload = _decode(authorization.removeprefix("Bearer ").strip())
    if payload.get("scope"):   # 영상 전용 토큰으로는 일반 API 를 못 쓴다
        raise _unauthorized("유효하지 않은 토큰입니다.")
    return payload


def user_from_payload(payload: dict, db: Session) -> "db_models.User":
    user = db.get(db_models.User, int(payload["sub"]))
    if user is None:
        raise _unauthorized("사용자를 찾을 수 없습니다.")
    return user


def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db),
) -> "db_models.User":
    """Authorization: Bearer <token> 헤더에서 현재 사용자를 해석한다."""
    return user_from_payload(bearer_payload(authorization), db)


def get_media_user(
    request: Request,
    authorization: str = Header(None),
    db: Session = Depends(get_db),
) -> "db_models.User":
    """영상·썸네일·클립 주소용 — 헤더가 있으면 헤더로, 없으면 로그인 때 받은 영상 쿠키로 확인한다."""
    if authorization:
        return get_current_user(authorization, db)
    token = request.cookies.get(MEDIA_COOKIE)
    if not token:
        raise _unauthorized("인증이 필요합니다.")
    payload = _decode(token)
    if payload.get("scope") != _MEDIA_SCOPE:
        raise _unauthorized("유효하지 않은 토큰입니다.")
    return user_from_payload(payload, db)


def _cookie_options(request: Request) -> dict:
    # 영상 주소(/api)에만 실리고, 자바스크립트로 못 읽고, 다른 사이트에서 온 요청엔 안 실린다.
    # HTTPS 로 접속했으면(nginx 뒤에선 X-Forwarded-Proto 로 판단) HTTPS 에서만 보낸다.
    return dict(path="/api", httponly=True, samesite="strict",
                secure=request.url.scheme == "https")


def set_media_cookie(response: Response, request: Request,
                     user: "db_models.User", expires_at: datetime) -> None:
    """일반 로그인 토큰과 같은 때 만료되는 영상 전용 토큰을 쿠키로 준다."""
    max_age = max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(MEDIA_COOKIE, create_token(user, _MEDIA_SCOPE, expires_at),
                        max_age=max_age, **_cookie_options(request))


def clear_media_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(MEDIA_COOKIE, **_cookie_options(request))
