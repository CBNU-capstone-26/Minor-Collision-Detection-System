"""회원가입 / 로그인 라우터."""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import (
    bearer_payload,
    burn_password_check,
    clear_media_cookie,
    create_token,
    get_current_user,
    hash_password,
    needs_rehash,
    set_media_cookie,
    user_from_payload,
    verify_password,
)
from app.settings import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=api_schemas.UserOut)
def signup(req: api_schemas.SignupRequest, db: Session = Depends(get_db)):
    if db.query(db_models.User).filter(
            db_models.User.username == req.username).first():
        raise HTTPException(status_code=400, detail="이미 사용 중인 아이디입니다.")
    if req.email and db.query(db_models.User).filter(
            db_models.User.email == req.email).first():
        raise HTTPException(status_code=400, detail="이미 사용 중인 이메일입니다.")

    user = db_models.User(
        username=req.username,
        password_hash=hash_password(req.password),
        name=req.name,
        email=req.email,
        role="USER",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=api_schemas.LoginResponse)
def login(req: api_schemas.LoginRequest, request: Request, response: Response,
          db: Session = Depends(get_db)):
    user = db.query(db_models.User).filter(
        db_models.User.username == req.username).first()
    if user is None:
        burn_password_check(req.password)
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )
    if needs_rehash(user.password_hash):   # 예전 저장 방식이면 이번에 새 방식으로 바꿔 둔다
        user.password_hash = hash_password(req.password)
        db.commit()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.AUTH_TOKEN_TTL_HOURS)
    set_media_cookie(response, request, user, expires_at)   # 영상·썸네일·클립용
    return api_schemas.LoginResponse(token=create_token(user, expires_at=expires_at), user=user)


@router.post("/session", response_model=api_schemas.UserOut)
def start_session(request: Request, response: Response,
                  authorization: str = Header(None), db: Session = Depends(get_db)):
    """화면을 새로 열 때 — 최신 사용자 정보를 주고, 영상 쿠키를 지금 로그인과 같은 만료로 다시 준다
    (쿠키가 지워졌거나 이 기능이 생기기 전에 로그인했어도 영상이 열리게)."""
    payload = bearer_payload(authorization)
    user = user_from_payload(payload, db)
    set_media_cookie(response, request, user, datetime.fromtimestamp(payload["exp"], timezone.utc))
    return user


@router.post("/logout")
def logout(request: Request, response: Response):
    """영상 쿠키를 지운다(로그인 토큰은 화면이 지운다)."""
    clear_media_cookie(response, request)
    return {"ok": True}


@router.get("/me", response_model=api_schemas.UserOut)
def get_me(current_user: db_models.User = Depends(get_current_user)):
    """현재 로그인한 사용자의 정보를 반환합니다."""
    return current_user


@router.put("/me", response_model=api_schemas.UserOut)
def update_me(
    req: api_schemas.UpdateUserRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """현재 로그인한 사용자의 프로필(이름, 이메일, 연락처)을 수정합니다."""
    if req.email and req.email != current_user.email:
        existing = db.query(db_models.User).filter(
            db_models.User.email == req.email,
            db_models.User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="이미 사용 중인 이메일 주소입니다.")
        current_user.email = req.email

    if req.name is not None:
        current_user.name = req.name.strip()
    if req.phone is not None:
        current_user.phone = req.phone.strip()

    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/change-password")
def change_password(
    req: api_schemas.ChangePasswordRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """현재 로그인한 사용자의 비밀번호를 변경합니다."""
    if not verify_password(req.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=400,
            detail="현재 비밀번호가 일치하지 않습니다.",
        )
    if len(req.new_password) < 4:
        raise HTTPException(
            status_code=400,
            detail="새 비밀번호는 최소 4자 이상이어야 합니다.",
        )
    current_user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"message": "비밀번호가 성공적으로 변경되었습니다."}
